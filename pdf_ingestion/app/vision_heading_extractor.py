"""
vision_heading_extractor.py — GPT-4o Vision heading extraction + PyMuPDF text assembly.

Strategy
--------
1. PyMuPDF renders each PDF page as a PNG image (fast, no model weights).
2. All pages are sent to GPT-4o Vision in overlapping batches of 10 (overlap = 2 pages)
   so a heading that starts at the end of one batch is always visible in the next.
3. GPT-4o returns ONLY top-level headings + start page for each batch. Results are
   merged and deduplicated across batches (same heading → keep lowest start page).
4. End pages are inferred: section[i].page_end = section[i+1].page_start.
   Shared boundary pages are given to BOTH neighbouring sections — no content
   is lost, slight duplication at boundaries is acceptable for RAG.
5. PyMuPDF page text is sliced by these page ranges to build PaperSection objects,
   using column-aware extraction (two-column IEEE/ACM papers handled correctly).
6. HeadingTree is built from the merged headings for the outline fast-path in QA.

Falls back gracefully:
  - If GPT-4o Vision returns nothing → empty SectionAssembly (caller falls back to Marker)
  - If a batch API call fails → that batch is skipped, others continue

Performance
-----------
  - PyMuPDF text extraction: ~1-3 seconds for any paper
  - GPT-4o Vision calls: parallel, ~5-15 seconds total regardless of paper length
  - Total: ~10-20 seconds vs Marker's 10+ minutes for born-digital PDFs
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import re
import tempfile
import os
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Batch config
# ---------------------------------------------------------------------------
BATCH_SIZE    = 10   # pages per GPT-4o Vision call
BATCH_OVERLAP = 2    # pages repeated at the boundary between batches
IMAGE_DPI     = 100  # lower DPI = smaller images = faster + cheaper API calls
                     # 100 DPI is sufficient for reading heading text

# ---------------------------------------------------------------------------
# GPT-4o Vision prompt
# ---------------------------------------------------------------------------
_HEADING_EXTRACTION_SYSTEM = """\
You are a precise document structure extractor.

Your task: extract ONLY the top-level section headings from the provided PDF pages,
and the page number where each heading STARTS (use the page numbers I give you, not
any printed page numbers on the paper itself).

Top-level headings are the MAIN sections of the paper such as:
  Abstract, Introduction, Related Work, Background, Methodology, Method, Approach,
  Experiments, Evaluation, Results, Discussion, Conclusion, Future Work, Limitations,
  References, Appendix, Acknowledgements

Top-level headings may be prefixed with Roman numerals (I, II, III, IV, V, VI, VII,
VIII, IX, X, XI, XII...) or Arabic numerals (1, 2, 3...). These prefixes are part of
the heading style, not separate content. For example:
  "XI. Conclusion" is a top-level heading with page = wherever "XI. Conclusion" appears
  "V. The Attack Surface of Artificial Intelligence" is also a top-level heading

CRITICAL PAGE ASSIGNMENT RULE:
- A heading belongs to the page where the heading TEXT itself is visible, even if it
  appears at the very bottom of that page with little or no body text following it.
- Do NOT assign a heading to the next page just because most of its content is there.
- If you see a heading at the bottom of page N, its page number is N.

DO NOT include:
  - Sub-headings or lettered subsections (A., B., C., 1.1, 2.3, A.1, etc.)
  - Figure captions or table titles
  - Author names, affiliations, paper titles
  - Any heading you are not confident is a real top-level section

Output format — respond with ONLY a JSON array, no other text:
[
  {"heading": "Introduction", "page": 2},
  {"heading": "Related Work", "page": 4},
  ...
]

If you find no top-level headings in these pages, return an empty array: []
"""

_HEADING_EXTRACTION_USER = """\
These are pages {start_page} to {end_page} of a research paper PDF.
Extract all top-level section headings visible in these pages and the page number
where each heading starts. Use the page numbers I told you ({start_page} to {end_page}),
not any printed numbers on the pages.

Remember: if a heading appears at the bottom of a page, assign it to THAT page number,
not the next one.
"""


# ---------------------------------------------------------------------------
# Page rendering (PyMuPDF → PNG bytes)
# ---------------------------------------------------------------------------

def _render_page_as_png(doc, page_idx: int, dpi: int = IMAGE_DPI, page_label: Optional[int] = None) -> Optional[bytes]:
    """Render a single PDF page to PNG bytes using PyMuPDF.

    If page_label is given, burns a visible "Page N" label into the top-left
    corner of the rendered image. This gives the vision model an unambiguous
    page number anchor so it can correctly assign headings that appear at the
    bottom of a page to THAT page rather than the next.
    """
    try:
        import fitz
        page = doc[page_idx]
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)

        if page_label is not None:
            # Draw a white rounded rectangle + bold black text label in top-left.
            # We use PIL if available (better text rendering); fall back to fitz annot.
            try:
                from PIL import Image, ImageDraw, ImageFont
                import io as _io
                img = Image.open(_io.BytesIO(pix.tobytes("png")))
                draw = ImageDraw.Draw(img)
                label = f" Page {page_label} "
                # Try to get a reasonable font size (≈2.5% of image height)
                font_size = max(14, int(img.height * 0.025))
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
                except Exception:
                    font = ImageFont.load_default()
                # Measure text
                bbox = draw.textbbox((0, 0), label, font=font)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                pad = 4
                # White background rectangle
                draw.rectangle([0, 0, tw + pad * 2, th + pad * 2], fill="white", outline="black")
                draw.text((pad, pad), label, fill="black", font=font)
                buf = _io.BytesIO()
                img.save(buf, format="PNG")
                return buf.getvalue()
            except ImportError:
                # PIL not available — write label as a fitz annotation (best-effort)
                pass  # fall through to plain PNG

        return pix.tobytes("png")
    except Exception as e:
        logger.warning("[VISION_HEADING] Page %d render failed: %s", page_idx + 1, e)
        return None


def _png_to_base64(png_bytes: bytes) -> str:
    return base64.b64encode(png_bytes).decode("utf-8")


# ---------------------------------------------------------------------------
# Single batch: call GPT-4o Vision for pages[start_idx:end_idx]
# ---------------------------------------------------------------------------

def _call_vision_batch(
    llm_client,
    doc,
    page_indices: List[int],   # 0-based indices into fitz doc
    page_numbers: List[int],   # 1-based page numbers matching page_indices
) -> List[Dict[str, Any]]:
    """
    Send a batch of PDF page images to GPT-4o Vision.
    Returns list of {"heading": str, "page": int} dicts.
    page numbers in the returned dicts are 1-based and match page_numbers param.
    """
    if not page_indices:
        return []

    start_page = page_numbers[0]
    end_page   = page_numbers[-1]

    # Build message content: text prompt + one image per page
    content: List[Dict] = [
        {
            "type": "text",
            "text": _HEADING_EXTRACTION_USER.format(
                start_page=start_page,
                end_page=end_page,
            ),
        }
    ]

    pages_added = 0
    for idx, pnum in zip(page_indices, page_numbers):
        png = _render_page_as_png(doc, idx, page_label=pnum)
        if png is None:
            continue
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{_png_to_base64(png)}",
                "detail": "low",   # "low" = cheaper + faster, sufficient for text layout
            },
        })
        pages_added += 1

    if pages_added == 0:
        logger.warning("[VISION_HEADING] No pages rendered for batch %d-%d", start_page, end_page)
        return []

    messages = [
        {"role": "system", "content": _HEADING_EXTRACTION_SYSTEM},
        {"role": "user",   "content": content},
    ]

    try:
        raw = llm_client.chat(messages=messages, temperature=0.0)
        if not raw or not raw.strip():
            return []

        # Strip markdown fences if GPT wraps in ```json ... ```
        cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.MULTILINE)
        cleaned = re.sub(r"\s*```$", "", cleaned.strip(), flags=re.MULTILINE)
        cleaned = cleaned.strip()

        parsed = json.loads(cleaned)
        if not isinstance(parsed, list):
            logger.warning("[VISION_HEADING] Unexpected response type: %s", type(parsed))
            return []

        results = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            heading = str(item.get("heading") or "").strip()
            page    = item.get("page")
            if not heading:
                continue
            try:
                page_int = int(page)
            except (TypeError, ValueError):
                continue
            # Validate page is within the batch range (GPT sometimes hallucinates)
            if start_page <= page_int <= end_page:
                results.append({
                    "heading": heading,
                    "page":    page_int,
                })

        logger.info(
            "[VISION_HEADING] Batch pages %d-%d → %d headings: %s",
            start_page, end_page, len(results),
            [(r["heading"], r["page"]) for r in results],
        )
        return results

    except json.JSONDecodeError as e:
        logger.warning(
            "[VISION_HEADING] JSON parse error for batch %d-%d: %s | raw=%r",
            start_page, end_page, e, raw[:200] if raw else "",
        )
        return []
    except Exception as e:
        logger.error("[VISION_HEADING] Batch %d-%d failed: %s", start_page, end_page, e)
        return []


# ---------------------------------------------------------------------------
# Merge + deduplicate heading results across batches
# ---------------------------------------------------------------------------

def _merge_batch_results(
    all_results: List[List[Dict[str, Any]]],
    total_pages: int,
) -> List[Dict[str, Any]]:
    """
    Merge heading detections from all batches.

    Deduplication rule: if the same heading text appears from multiple batches
    (due to overlap), keep the one with the LOWEST start page.

    Returns list of:
      {"heading": str, "page_start": int, "page_end": int}
    sorted by page_start ascending.
    """
    # Collect all detections, normalise heading text for deduplication.
    # Normalisation: lowercase + strip section numbering ("1.", "I.", "A." prefixes)
    seen:      Dict[str, int] = {}  # normalised_key → lowest page_start seen
    canonical: Dict[str, str] = {}  # normalised_key → original heading text

    for batch in all_results:
        for item in batch:
            heading = item["heading"].strip()
            page    = item["page"]
            key = re.sub(r"^[\d]+[.)]\s*|^[IVXivx]+[.)]\s*", "", heading.lower()).strip()
            key = re.sub(r"\s+", " ", key)
            if key not in seen or page < seen[key]:
                seen[key]      = page
                canonical[key] = heading   # prefer text from earliest occurrence

    if not seen:
        return []

    # Build sorted list
    merged = [
        {"heading": canonical[k], "page_start": seen[k]}
        for k in seen
        if k in canonical
    ]
    merged.sort(key=lambda x: x["page_start"])

    # Infer page_end for each section.
    # When a heading starts on page N, the previous section's page_end = N.
    # Both sections share that page — the previous section gets it because its
    # content may run to the middle of that page, and the next section gets it
    # because its heading starts there. This is intentional duplication: no
    # content is lost, both sections are fully represented.
    # e.g. Abstract pages 1→1, Introduction pages 1→3, Background pages 3→6
    for i, section in enumerate(merged):
        if i + 1 < len(merged):
            section["page_end"] = merged[i + 1]["page_start"]
        else:
            section["page_end"] = total_pages

    return merged


# ---------------------------------------------------------------------------
# PyMuPDF-based heading page anchoring (post-vision correction)
# ---------------------------------------------------------------------------

def _anchor_headings_with_pymupdf(
    merged: List[Dict[str, Any]],
    doc,
    total_pages: int,
    search_window: int = 2,
) -> List[Dict[str, Any]]:
    """
    Use PyMuPDF font-aware span search to pin each heading to the correct page.

    WHY NAIVE TEXT SEARCH FAILS
    ----------------------------
    page.search_for("Introduction") matches every occurrence of the word — section
    headings AND inline mentions ("as described in the Introduction..."). To find
    only the real heading we use four layered signals:

      1. SEARCH WINDOW (primary guard against cross-section contamination)
         Each heading is only searched in [vision_page - search_window, vision_page + 1].
         Because vision's estimate is already in the right neighbourhood (±1-2 pages),
         a heading on page 33 will never accidentally match a mention of "Methodology"
         in the Introduction on page 3 — those pages are outside the window entirely.

      2. SECTION-NUMBER PREFIX (most discriminative signal)
         If the heading carries a Roman/Arabic prefix ("XI. Conclusion", "3. Methodology"),
         we search for the FULL prefixed string first. Prefixes virtually never appear
         in body-text mentions, so a match is definitively the real heading.
         Only when the prefix search fails do we fall through to font-size heuristics.

      3. FONT SIZE >= BODY BASELINE (rejects body-text mentions in same window)
         We compute the modal (most-frequent) font size on the page as the body-text
         baseline. A span must be >= 95% of that size to qualify. Inline mentions of
         heading words share the body font; real headings are the same size or larger.

      4. RUNNING-HEADER / FOOTER EXCLUSION (rejects page-header repetitions)
         Running headers that repeat the section name sit at the very top or bottom
         of the page (y-position < 8% or > 92% of page height). We skip those zones
         so "Introduction" printed as a chapter header on pages 2-5 does not steal
         page 1's heading assignment.

      5. STANDALONE LINE (rejects mid-sentence occurrences)
         A real heading occupies its own line. We require the total text in the matched
         line to be no more than 20 characters longer than the heading itself, which
         rejects spans that appear in the middle of a long sentence.

    Algorithm
    ---------
    For each section heading (in document order):
      1. Narrow the search to [vision_page - search_window, vision_page + 1].
      2. On each page in that window call _find_heading_span() which applies signals
         2-5 in order, returning True on the first qualifying match.
      3. The first page in the window with a qualifying match becomes page_start.
      4. No match → keep vision's page_start (safe fallback, never makes things worse).
    After all corrections: re-sort and rebuild page_end values.
    """
    from collections import Counter

    def _norm(s: str) -> str:
        s = _expand_ligatures(s.lower())
        return re.sub(r"\s+", " ", s).strip()

    def _strip_prefix(h: str) -> str:
        return re.sub(r"^[\d]+[.)]\s*|^[IVXivx]+[.)]\s*", "", h.strip()).strip()

    def _has_prefix(h: str) -> bool:
        return bool(re.match(r"^[\d]+[.)]\s*|^[IVXivx]+[.)]\s*", h.strip()))

    def _modal_font_size(page_dict: dict) -> float:
        """Modal font size = body-text baseline for this page."""
        sizes: List[float] = []
        for block in page_dict.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    if span.get("text", "").strip():
                        sizes.append(round(span.get("size", 0), 1))
        if not sizes:
            return 10.0
        return Counter(sizes).most_common(1)[0][0]

    def _in_header_footer_zone(line: dict, page_height: float) -> bool:
        """True if line sits in the top-8% or bottom-8% margin (running header/footer)."""
        bbox = line.get("bbox")  # (x0, y0, x1, y1)
        if not bbox or page_height <= 0:
            return False
        y0, y1 = bbox[1], bbox[3]
        mid_y = (y0 + y1) / 2.0
        return mid_y < page_height * 0.08 or mid_y > page_height * 0.92

    def _find_heading_span(
        page,
        page_dict: dict,
        heading_text: str,
        body_size: float,
    ) -> bool:
        """
        Return True if this page has a heading-quality occurrence of heading_text.

        Signal 2 — prefixed full string (skip body-text mentions entirely).
        Signal 3 + 4 + 5 — font size, header/footer exclusion, standalone line.
        """
        ht_full = heading_text.strip()
        ht_stripped = _strip_prefix(ht_full)
        has_pfx = _has_prefix(ht_full)
        page_height = page.rect.height

        # Signal 2: search for full prefixed string — match = definitely the heading
        if has_pfx:
            for candidate in [
                ht_full, _expand_ligatures(ht_full),
                ht_full.lower(), _expand_ligatures(ht_full).lower(),
            ]:
                if page.search_for(candidate, quads=False):
                    return True

        # Signals 3+4+5: walk spans for the stripped heading name
        target_norm = _norm(ht_stripped)
        if not target_norm:
            return False

        for block in page_dict.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                # Signal 4: skip running headers / footers
                if _in_header_footer_zone(line, page_height):
                    continue

                spans = line.get("spans", [])
                line_norm = _norm("".join(s.get("text", "") for s in spans))

                if target_norm not in line_norm:
                    continue

                for span in spans:
                    span_norm = _norm(span.get("text", ""))
                    if target_norm not in span_norm and span_norm not in target_norm:
                        continue

                    # Signal 3: font size >= 95% of body baseline
                    if span.get("size", 0) < body_size * 0.95:
                        continue

                    # Signal 5: heading should dominate its line (<=20 extra chars)
                    if len(line_norm) - len(target_norm) > 20:
                        continue

                    return True

        return False

    # ── Main anchoring loop ───────────────────────────────────────────────────
    corrected = 0
    for sec in merged:
        vision_page = sec["page_start"]
        lo = max(1, vision_page - search_window)
        hi = min(total_pages, vision_page + 1)

        best_page: Optional[int] = None
        for pnum in range(lo, hi + 1):
            try:
                page = doc[pnum - 1]
                page_dict = page.get_text("dict", flags=0)
                body_size = _modal_font_size(page_dict)
                if _find_heading_span(page, page_dict, sec["heading"], body_size):
                    best_page = pnum
                    break
            except Exception as exc:
                logger.debug("[VISION_HEADING] Anchor error page %d: %s", pnum, exc)

        if best_page is not None and best_page != vision_page:
            logger.info(
                "[VISION_HEADING] Anchor corrected '%s': page %d -> %d",
                sec["heading"], vision_page, best_page,
            )
            sec["page_start"] = best_page
            corrected += 1

    if corrected:
        logger.info(
            "[VISION_HEADING] PyMuPDF anchoring corrected %d/%d heading(s)",
            corrected, len(merged),
        )
        merged.sort(key=lambda x: x["page_start"])
        for i, section in enumerate(merged):
            if i + 1 < len(merged):
                section["page_end"] = merged[i + 1]["page_start"]
            else:
                section["page_end"] = total_pages

    return merged


# ---------------------------------------------------------------------------
# Shared-page text splitter
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Ligature normalisation table
# ---------------------------------------------------------------------------
# LaTeX PDFs store ligatures as single Unicode codepoints (e.g. ﬁ = fi).
# PyMuPDF returns them as-is; GPT-4o Vision reads the rendered image and
# returns normal ASCII letters. This table maps ligature → ASCII expansion
# so both strings can be compared after normalisation.
_LIGATURES: Dict[str, str] = {
    "\ufb00": "ff",   # ﬀ
    "\ufb01": "fi",   # ﬁ
    "\ufb02": "fl",   # ﬂ
    "\ufb03": "ffi",  # ﬃ
    "\ufb04": "ffl",  # ﬄ
    "\ufb05": "st",   # ﬅ
    "\ufb06": "st",   # ﬆ
    "\u0133": "ij",   # ĳ
    "\u0132": "IJ",   # Ĳ
}

def _expand_ligatures(s: str) -> str:
    """Replace Unicode ligature characters with their ASCII equivalents."""
    for lig, exp in _LIGATURES.items():
        s = s.replace(lig, exp)
    return s




# ---------------------------------------------------------------------------
# Build SectionAssembly + HeadingTree from merged headings + PyMuPDF page text
# ---------------------------------------------------------------------------

def _build_sections_from_headings(
    merged_headings: List[Dict[str, Any]],
    page_texts: Dict[int, str],    # 1-based page_num → extracted text
    total_pages: int,
) -> Tuple["SectionAssembly", "HeadingTree"]:
    """
    Combine GPT-4o heading positions with PyMuPDF page text to produce
    PaperSection objects and a HeadingTree.

    Shared pages are given to BOTH neighbouring sections intentionally.
    e.g. if Introduction starts mid-page-2:
      Abstract    → pages 1–2  (includes tail of page 2)
      Introduction → pages 2–4 (includes head of page 2)
    No content is lost. The slight duplication at boundaries is acceptable
    for RAG — both sections are fully represented.

    A one-page backwards buffer (page_start - 1) is also applied so that
    headings detected one page late (bottom-of-page misattribution) still
    capture the correct content. The LLM ignores irrelevant overlap.
    """
    from pdf_ingestion.app.section_assembler import SectionAssembly, PaperSection
    from pdf_ingestion.app.structural_extractor import HeadingTree, Heading

    sections: List[PaperSection] = []
    headings: List[Heading]      = []

    for idx, sec in enumerate(merged_headings):
        heading_text   = sec["heading"]
        page_start     = sec["page_start"]
        page_end       = sec["page_end"]

        # One-page backwards buffer: if the heading was detected one page late
        # (bottom-of-page misattribution), this ensures the actual heading page
        # is still included in the section's content. The LLM filters any noise.
        fetch_from = max(1, page_start - 1)

        text_parts: List[str] = []
        for pnum in range(fetch_from, page_end + 1):
            raw_page_text = (page_texts.get(pnum) or "").strip()
            if raw_page_text:
                text_parts.append(raw_page_text)

        content_text = "\n\n".join(text_parts).strip()

        sections.append(PaperSection(
            section_index  = idx,
            heading_level  = 1,
            heading_text   = heading_text,
            parent_heading = None,
            page_start     = page_start,
            page_end       = page_end,
            content_text   = content_text,
            content_length = len(content_text),
        ))

        headings.append(Heading(
            level = 1,
            text  = heading_text,
            page  = page_start,
        ))

    # Filter out sections with no content and re-index
    sections = [s for s in sections if s.content_text.strip()]
    for i, s in enumerate(sections):
        s.section_index = i

    return SectionAssembly(sections=sections), HeadingTree(headings=headings)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def extract_sections_via_vision_headings(
    pdf_bytes: bytes,
    llm_client,
    settings=None,
) -> Tuple["SectionAssembly", "HeadingTree", int]:
    """
    Extract paper sections using GPT-4o Vision for heading detection
    and PyMuPDF for text extraction.

    Pipeline:
      1. PyMuPDF: extract flat page text + render page images
      2. GPT-4o Vision: detect top-level headings + start pages (parallel batches)
      3. Merge + deduplicate heading detections across overlapping batches
      4. Slice PyMuPDF page text by heading page ranges → PaperSection objects
      5. Build HeadingTree for outline fast-path in qa_service

    Returns (SectionAssembly, HeadingTree, total_pages).
    On any failure returns (SectionAssembly(error=...), HeadingTree(error=...), 0).
    """
    from pdf_ingestion.app.section_assembler import SectionAssembly
    from pdf_ingestion.app.structural_extractor import HeadingTree

    def _err(msg: str):
        logger.error("[VISION_HEADING] %s", msg)
        return SectionAssembly(error=msg), HeadingTree(error=msg), 0

    if not pdf_bytes:
        return _err("Empty PDF bytes.")

    if llm_client is None or not llm_client.enabled():
        return _err("LLM client not available for Vision heading extraction.")

    # ── 1. Open PDF with PyMuPDF ──────────────────────────────────────────────
    try:
        import fitz
    except ImportError:
        return _err("PyMuPDF (fitz) not installed. Run: pip install pymupdf")

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        return _err(f"Could not open PDF: {e}")

    total_pages = len(doc)
    if total_pages == 0:
        doc.close()
        return _err("PDF has zero pages.")

    # ── 2. Extract page text via PyMuPDF (fast, ~1-3 sec total) ──────────────
    #
    # Column-aware extraction strategy:
    # ─────────────────────────────────
    # PyMuPDF's default get_text("text") reads blocks in PDF-storage order, which
    # for two-column papers (IEEE, ACM, medical journals) causes left/right column
    # text to be interleaved — breaking chunking and RAG quality.
    #
    # Fix: use get_text("dict") to get bounding boxes, detect two-column layout
    # via a gap-based heuristic, then sort blocks into correct reading order.
    #
    # Column detection (conservative — only fires when clearly two-column):
    #   1. Exclude full-width blocks (spanning >80% of page) — these are titles.
    #   2. Collect the unique x0 (left-edge) values of all remaining blocks.
    #   3. Find the largest gap between consecutive x0 values.
    #   4. Two-column if: gap > 20% of page width AND gap midpoint is in the
    #      middle 25–75% of the page AND both sides have >= 2 blocks.
    #
    # Single-column papers (most arXiv ML/CS preprints): heuristic never fires.
    # Falls back to get_text("text") if dict extraction errors on any page.

    def _detect_and_extract(page) -> Tuple[str, bool]:
        """
        Returns (text, is_two_column).
        Extracts text with correct reading order for both 1- and 2-column layouts.
        """
        try:
            raw = page.get_text("dict", flags=0)
        except Exception:
            return page.get_text("text", flags=0).strip(), False

        pw = page.rect.width
        if pw <= 0:
            return page.get_text("text", flags=0).strip(), False

        blocks = [b for b in raw.get("blocks", []) if b.get("type") == 0]
        if not blocks:
            return "", False

        # Full-width blocks span the entire page width (titles, abstract headers)
        full_blks = [b for b in blocks if (b["bbox"][2] - b["bbox"][0]) > pw * 0.80]
        full_ids  = {id(b) for b in full_blks}
        non_full  = [b for b in blocks if id(b) not in full_ids]

        # ── Gap-based column detection ─────────────────────────────────────────
        # Collect distinct x0 positions of non-full-width blocks.
        # A large gap in x0 values with the gap midpoint in the page's middle zone
        # is the signature of a two-column layout.
        is_two_col = False
        divider    = pw / 2.0   # default fallback

        if len(non_full) >= 4:
            x0s = sorted({round(b["bbox"][0]) for b in non_full})
            if len(x0s) >= 2:
                # Find (gap_size, gap_midpoint) for each consecutive x0 pair
                gap_pairs = [
                    (x0s[i + 1] - x0s[i], (x0s[i] + x0s[i + 1]) / 2.0)
                    for i in range(len(x0s) - 1)
                ]
                best_gap, best_mid = max(gap_pairs, key=lambda g: g[0])

                # Fire only if: gap is large AND in the middle of the page
                if (best_gap > pw * 0.20
                        and pw * 0.25 <= best_mid <= pw * 0.75):
                    left_blks  = [b for b in non_full if b["bbox"][0] < best_mid]
                    right_blks = [b for b in non_full if b["bbox"][0] >= best_mid]
                    if len(left_blks) >= 2 and len(right_blks) >= 2:
                        is_two_col = True
                        divider    = best_mid

        # ── Sort blocks into reading order ────────────────────────────────────
        if is_two_col:
            col_left  = sorted([b for b in non_full if b["bbox"][0] <  divider], key=lambda b: b["bbox"][1])
            col_right = sorted([b for b in non_full if b["bbox"][0] >= divider], key=lambda b: b["bbox"][1])
            full_blks.sort(key=lambda b: b["bbox"][1])
            # Reading order: full-width headers → left column top-to-bottom → right column top-to-bottom
            ordered = full_blks + col_left + col_right
        else:
            ordered = sorted(blocks, key=lambda b: (b["bbox"][1], b["bbox"][0]))

        # ── Assemble text ──────────────────────────────────────────────────────
        lines: List[str] = []
        for blk in ordered:
            for line in blk.get("lines", []):
                line_text = "".join(s.get("text", "") for s in line.get("spans", [])).strip()
                if line_text:
                    lines.append(line_text)
        return "\n".join(lines).strip(), is_two_col

    page_texts: Dict[int, str] = {}
    two_col_pages = 0

    for page_idx in range(total_pages):
        try:
            page = doc[page_idx]
            text, is_two_col = _detect_and_extract(page)
            if text:
                page_texts[page_idx + 1] = text   # 1-based key
            if is_two_col:
                two_col_pages += 1
        except Exception as e:
            logger.warning("[VISION_HEADING] Page %d text extraction failed: %s", page_idx + 1, e)

    if not page_texts:
        doc.close()
        return _err("PyMuPDF found no embedded text — PDF may be scanned.")

    layout_guess = "two-column" if two_col_pages > total_pages * 0.3 else "single-column"
    logger.info(
        "[VISION_HEADING] PyMuPDF extracted text from %d/%d pages "
        "| layout=%s (%d/%d pages two-column)",
        len(page_texts), total_pages, layout_guess, two_col_pages, total_pages,
    )

    # ── 3. Build overlapping batches ──────────────────────────────────────────
    # pages are 1-based throughout
    batches: List[List[int]] = []  # each element is a list of 1-based page numbers
    start = 1
    while start <= total_pages:
        end = min(start + BATCH_SIZE - 1, total_pages)
        batches.append(list(range(start, end + 1)))
        if end == total_pages:
            break
        # Next batch starts BATCH_OVERLAP pages before this batch ends
        start = end - BATCH_OVERLAP + 1

    logger.info(
        "[VISION_HEADING] Sending %d batches to GPT-4o Vision (batch_size=%d, overlap=%d)",
        len(batches), BATCH_SIZE, BATCH_OVERLAP,
    )

    # ── 4. Call GPT-4o Vision for each batch (run concurrently via threads) ───
    # We use ThreadPoolExecutor because llm_client.chat() is synchronous.
    # asyncio.get_running_loop().run_in_executor would require this function
    # to be async; since it's called from an executor already (see ingest_pipeline),
    # we just use concurrent.futures directly here.
    import concurrent.futures

    batch_results: List[List[Dict[str, Any]]] = [[] for _ in batches]

    def _run_batch(batch_idx: int, page_numbers: List[int]) -> List[Dict[str, Any]]:
        page_indices = [p - 1 for p in page_numbers]  # convert to 0-based for fitz
        return _call_vision_batch(
            llm_client  = llm_client,
            doc         = doc,
            page_indices= page_indices,
            page_numbers= page_numbers,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(batches), 5)) as pool:
        futures = {
            pool.submit(_run_batch, i, batch): i
            for i, batch in enumerate(batches)
        }
        for future in concurrent.futures.as_completed(futures):
            i = futures[future]
            try:
                batch_results[i] = future.result()
            except Exception as e:
                logger.error("[VISION_HEADING] Batch %d raised exception: %s", i, e)
                batch_results[i] = []

    # ── 5. Merge + deduplicate headings across batches ────────────────────────
    merged = _merge_batch_results(batch_results, total_pages)

    # ── 5b. PyMuPDF anchoring: correct off-by-one page errors from vision ─────
    # The vision model sometimes assigns a heading to the page AFTER where the
    # heading text physically appears (bottom-of-page effect). We use PyMuPDF's
    # exact text search to pin each heading to its true page. doc is still open.
    if merged:
        merged = _anchor_headings_with_pymupdf(merged, doc, total_pages)

    doc.close()

    if not merged:
        return _err(
            "GPT-4o Vision found no top-level headings in the PDF. "
            "The paper may use an unusual layout."
        )

    logger.info(
        "[VISION_HEADING] Final merged headings (%d): %s",
        len(merged),
        [(m["heading"], m["page_start"], m["page_end"]) for m in merged],
    )

    # ── 6. Build SectionAssembly + HeadingTree ────────────────────────────────
    assembly, heading_tree = _build_sections_from_headings(
        merged_headings = merged,
        page_texts      = page_texts,
        total_pages     = total_pages,
    )

    if assembly.is_empty():
        return _err("Sections were detected but all had empty content after text assembly.")

    logger.info(
        "[VISION_HEADING] Complete: %d sections, %d headings",
        len(assembly.sections), len(heading_tree.headings),
    )

    return assembly, heading_tree, total_pages