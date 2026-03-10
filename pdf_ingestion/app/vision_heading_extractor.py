"""
vision_heading_extractor.py — GPT-4o Vision heading extraction + PyMuPDF text assembly.

Strategy
--------
1. PyMuPDF renders each PDF page as a PNG image (fast, no model weights).
2. All pages are sent to GPT-4o Vision in overlapping batches of 10 (overlap = 2 pages)
   so a heading that starts at the end of one batch is always visible in the next.
3. GPT-4o returns ONLY top-level headings + start page for each batch. Results are
   merged and deduplicated across batches (same heading → keep lowest start page).
4. A focused early-pages pass (pages 1-2 by default) is also run at higher quality,
   because page 1 is usually the densest page in research PDFs.
5. End pages are inferred: section[i].page_end = section[i+1].page_start.
   Shared boundary pages are given to BOTH neighbouring sections — no content
   is lost, slight duplication at boundaries is acceptable for RAG.
6. PyMuPDF page text is sliced by these page ranges to build PaperSection objects,
   using column-aware extraction (two-column IEEE/ACM papers handled correctly).
7. HeadingTree is built from the merged headings for the outline fast-path in QA.

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

import base64
import json
import logging
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Batch config
# ---------------------------------------------------------------------------
BATCH_SIZE = 10
BATCH_OVERLAP = 2

# Standard render / vision settings
IMAGE_DPI = 100
IMAGE_DETAIL = "low"

# Better quality for the first normal batch (the one containing page 1)
FIRST_BATCH_IMAGE_DPI = 160
FIRST_BATCH_IMAGE_DETAIL = "high"

# Extra focused pass for the first few pages of every PDF
EARLY_PAGES_FOCUSED_COUNT = 2
EARLY_PAGES_IMAGE_DPI = 180
EARLY_PAGES_IMAGE_DETAIL = "high"

# ---------------------------------------------------------------------------
# GPT-4o Vision prompt
# ---------------------------------------------------------------------------
_HEADING_EXTRACTION_SYSTEM = """\
You are a precise document-structure extractor.

Task:
Extract ONLY top-level section headings that are visibly present on the provided PDF pages,
and the page number where each heading STARTS.

Use the page numbers supplied in the prompt, NOT any printed page numbers shown inside the PDF.

What counts as a top-level heading:
- Main paper sections such as:
  Abstract, Introduction, Background, Related Work, Prior Research, Methodology,
  Method, Approach, Experiments, Evaluation, Results, Discussion, Conclusion,
  Future Work, Limitations, References, Bibliography, Appendix, Acknowledgements.
- These may appear in different styles:
  - Unnumbered: "Abstract", "References"
  - Arabic-numbered: "1. Introduction", "2 Method"
  - Roman-numbered: "I. Introduction", "II. Prior Research"
  - Mixed styles within the same paper
- A heading may appear on the same page as title/authors/affiliations/keywords.

IMPORTANT:
- Page 1 may contain MORE THAN ONE top-level heading.
- Do NOT stop after finding only one heading on a page.
- Scan the ENTIRE page, including content below affiliations, footnotes, and keywords.
- If a heading appears at the bottom of a page, assign it to THAT page.
- A single "I." may be a real Roman numeral heading prefix. Do not assume it is a footnote marker.

ALWAYS include these when they are visibly section headings:
- Abstract
- Acknowledgements / Acknowledgments
- References / Bibliography
- Appendix / Appendices

DO NOT include:
- The paper title
- Author names, affiliations, emails, keywords
- Figure captions or table captions
- Subsection headings such as:
  1.1, 2.3, III-A, A., B., C., A.1, etc.
- Running headers / page headers / footer text

Decision rule:
Return something only if it visually looks like a MAIN section heading.
Do not invent headings that are not visible.

Output format:
Return ONLY a JSON array, no extra text:
[
  {"heading": "Abstract", "page": 1},
  {"heading": "I. Introduction", "page": 1},
  {"heading": "II. Prior Research", "page": 7}
]

If no top-level headings are visible in these pages, return:
[]
"""

_HEADING_EXTRACTION_USER = """\
These are pages {start_page} to {end_page} of a research paper PDF.
Extract all top-level section headings visible in these pages and the page number
where each heading starts.

Use the page numbers I told you ({start_page} to {end_page}), not any printed page numbers in the PDF.

Important reminders:
- Top-level headings may be unnumbered, Roman-numbered, Arabic-numbered, or mixed.
- Page 1 may contain multiple top-level headings.
- If a heading is visible at the bottom of a page, assign it to THAT page.
- Do not stop after finding only one heading on a page.
- Ignore title, authors, affiliations, emails, keywords, captions, and subsection headings.
- Return only headings that visually look like MAIN section headings.
"""

# ---------------------------------------------------------------------------
# Patterns / labels
# ---------------------------------------------------------------------------

_ROMAN_PREFIX_RE = re.compile(r"^\s*([IVXivx]+)[.)]\s+")
_ARABIC_PREFIX_RE = re.compile(r"^\s*(\d+)[.)]\s+")
_SUBSECTION_DECIMAL_RE = re.compile(r"^\s*\d+\.\d+")
_ALPHA_PREFIX_RE = re.compile(r"^\s*[A-Z][.)]\s+")
_ROMAN_HYPHEN_SUB_RE = re.compile(r"^\s*[IVXivx]+[-–—][A-Z0-9]+")
_APPENDIX_RE = re.compile(r"^\s*appendix(?:\s+[A-Z0-9]+)?\b", re.IGNORECASE)

_MANDATORY_UNNUMBERED_HEADINGS = {
    "abstract",
    "acknowledgements",
    "acknowledgments",
    "references",
    "reference",
    "bibliography",
    "appendix",
    "appendices",
    "conclusion",
    "conclusions",
}

_COMMON_SECTION_WORDS = {
    "abstract",
    "introduction",
    "background",
    "related work",
    "prior research",
    "research methodology",
    "methodology",
    "method",
    "approach",
    "experiments",
    "experimental setup",
    "evaluation",
    "results",
    "discussion",
    "conclusion",
    "conclusions",
    "future work",
    "limitations",
    "references",
    "reference",
    "bibliography",
    "appendix",
    "appendices",
    "acknowledgements",
    "acknowledgments",
}

_ROMAN_TO_INT: Dict[str, int] = {
    "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5,
    "VI": 6, "VII": 7, "VIII": 8, "IX": 9, "X": 10,
    "XI": 11, "XII": 12, "XIII": 13, "XIV": 14, "XV": 15,
    "XVI": 16, "XVII": 17, "XVIII": 18, "XIX": 19, "XX": 20,
}
_INT_TO_ROMAN: Dict[int, str] = {v: k for k, v in _ROMAN_TO_INT.items()}

# ---------------------------------------------------------------------------
# Ligature normalisation table
# ---------------------------------------------------------------------------
_LIGATURES: Dict[str, str] = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\ufb05": "st",
    "\ufb06": "st",
    "\u0133": "ij",
    "\u0132": "IJ",
}


def _expand_ligatures(s: str) -> str:
    for lig, exp in _LIGATURES.items():
        s = s.replace(lig, exp)
    return s


def _norm(s: str) -> str:
    s = _expand_ligatures(s.lower())
    return re.sub(r"\s+", " ", s).strip()


def _strip_prefix(h: str) -> str:
    h = re.sub(r"^\s*[IVXivx]+[.)]\s*", "", h)
    h = re.sub(r"^\s*\d+[.)]\s*", "", h)
    return h.strip()


def _heading_prefix_kind(h: str) -> str:
    text = h.strip()

    if _SUBSECTION_DECIMAL_RE.match(text):
        return "decimal-subsection"
    if _ROMAN_HYPHEN_SUB_RE.match(text):
        return "roman-hyphen-subsection"
    if _APPENDIX_RE.match(text):
        return "appendix"
    if _ROMAN_PREFIX_RE.match(text):
        return "roman"
    if _ARABIC_PREFIX_RE.match(text):
        return "arabic"
    if _ALPHA_PREFIX_RE.match(text):
        return "alpha-subsection"

    stripped = _strip_prefix(text)
    if _norm(stripped) in _MANDATORY_UNNUMBERED_HEADINGS:
        return "mandatory-unnumbered"

    return "unnumbered"


def _is_subsection_like_heading(h: str) -> bool:
    kind = _heading_prefix_kind(h)
    return kind in {"decimal-subsection", "roman-hyphen-subsection", "alpha-subsection"}


def _dominant_top_level_scheme(headings: List[Dict[str, Any]]) -> str:
    counts = Counter()
    for sec in headings:
        kind = _heading_prefix_kind(sec["heading"])
        if kind in {"roman", "arabic", "appendix"}:
            counts[kind] += 1
    if not counts:
        return "unknown"
    dominant, n = counts.most_common(1)[0]
    return dominant if n >= 2 else "unknown"


# ---------------------------------------------------------------------------
# Page rendering (PyMuPDF → PNG bytes)
# ---------------------------------------------------------------------------

def _render_page_as_png(
    doc,
    page_idx: int,
    dpi: int = IMAGE_DPI,
    page_label: Optional[int] = None,
) -> Optional[bytes]:
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
            try:
                from PIL import Image, ImageDraw, ImageFont
                import io as _io

                img = Image.open(_io.BytesIO(pix.tobytes("png")))
                draw = ImageDraw.Draw(img)
                label = f" Page {page_label} "
                font_size = max(14, int(img.height * 0.025))
                try:
                    font = ImageFont.truetype(
                        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                        font_size,
                    )
                except Exception:
                    font = ImageFont.load_default()

                bbox = draw.textbbox((0, 0), label, font=font)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                pad = 4
                draw.rectangle(
                    [0, 0, tw + pad * 2, th + pad * 2],
                    fill="white",
                    outline="black",
                )
                draw.text((pad, pad), label, fill="black", font=font)
                buf = _io.BytesIO()
                img.save(buf, format="PNG")
                return buf.getvalue()
            except ImportError:
                pass

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
    page_indices: List[int],
    page_numbers: List[int],
    *,
    dpi: int = IMAGE_DPI,
    detail: str = IMAGE_DETAIL,
    batch_tag: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Send a batch of PDF page images to GPT-4o Vision.
    Returns list of {"heading": str, "page": int} dicts.
    """
    if not page_indices:
        return []

    start_page = page_numbers[0]
    end_page = page_numbers[-1]
    tag = batch_tag or f"{start_page}-{end_page}"

    content: List[Dict[str, Any]] = [
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
        png = _render_page_as_png(doc, idx, dpi=dpi, page_label=pnum)
        if png is None:
            continue

        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{_png_to_base64(png)}",
                    "detail": detail,
                },
            }
        )
        pages_added += 1

    if pages_added == 0:
        logger.warning("[VISION_HEADING] No pages rendered for batch %s", tag)
        return []

    messages = [
        {"role": "system", "content": _HEADING_EXTRACTION_SYSTEM},
        {"role": "user", "content": content},
    ]

    try:
        raw = llm_client.chat(messages=messages, temperature=0.0)
        if not raw or not raw.strip():
            return []

        cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.MULTILINE)
        cleaned = re.sub(r"\s*```$", "", cleaned.strip(), flags=re.MULTILINE)
        cleaned = cleaned.strip()

        parsed = json.loads(cleaned)
        if not isinstance(parsed, list):
            logger.warning("[VISION_HEADING] Unexpected response type for batch %s: %s", tag, type(parsed))
            return []

        results: List[Dict[str, Any]] = []
        for item in parsed:
            if not isinstance(item, dict):
                continue

            heading = str(item.get("heading") or "").strip()
            page = item.get("page")
            if not heading:
                continue

            try:
                page_int = int(page)
            except (TypeError, ValueError):
                continue

            if start_page <= page_int <= end_page:
                results.append({"heading": heading, "page": page_int})

        logger.info(
            "[VISION_HEADING] Batch %s → %d headings (dpi=%d, detail=%s): %s",
            tag,
            len(results),
            dpi,
            detail,
            [(r["heading"], r["page"]) for r in results],
        )
        return results

    except json.JSONDecodeError as e:
        logger.warning(
            "[VISION_HEADING] JSON parse error for batch %s: %s | raw=%r",
            tag,
            e,
            raw[:200] if raw else "",
        )
        return []
    except Exception as e:
        logger.error("[VISION_HEADING] Batch %s failed: %s", tag, e)
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
    seen: Dict[str, int] = {}
    canonical: Dict[str, str] = {}

    for batch in all_results:
        for item in batch:
            heading = item["heading"].strip()
            page = item["page"]

            if _is_subsection_like_heading(heading):
                logger.info("[VISION_HEADING] Dropping subsection-like heading at merge: %s", heading)
                continue

            key = _norm(_strip_prefix(heading))
            if not key:
                continue

            if key not in seen or page < seen[key]:
                seen[key] = page
                canonical[key] = heading

    if not seen:
        return []

    merged = [
        {"heading": canonical[k], "page_start": seen[k]}
        for k in seen
        if k in canonical
    ]
    merged.sort(key=lambda x: (x["page_start"], _norm(x["heading"])))

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
    """

    def _modal_font_size(page_dict: dict) -> float:
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
        bbox = line.get("bbox")
        if not bbox or page_height <= 0:
            return False
        y0, y1 = bbox[1], bbox[3]
        mid_y = (y0 + y1) / 2.0
        return mid_y < page_height * 0.08 or mid_y > page_height * 0.92

    def _find_heading_span(page, page_dict: dict, heading_text: str, body_size: float) -> bool:
        ht_full = heading_text.strip()
        ht_stripped = _strip_prefix(ht_full)
        has_prefixed_number = bool(_ROMAN_PREFIX_RE.match(ht_full) or _ARABIC_PREFIX_RE.match(ht_full))
        page_height = page.rect.height

        if has_prefixed_number:
            for candidate in [
                ht_full,
                _expand_ligatures(ht_full),
                ht_full.lower(),
                _expand_ligatures(ht_full).lower(),
            ]:
                if page.search_for(candidate, quads=False):
                    return True

        target_norm = _norm(ht_stripped)
        if not target_norm:
            return False

        for block in page_dict.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
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

                    if span.get("size", 0) < body_size * 0.95:
                        continue

                    if len(line_norm) - len(target_norm) > 20:
                        continue

                    return True

        return False

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
                sec["heading"],
                vision_page,
                best_page,
            )
            sec["page_start"] = best_page
            corrected += 1

    if corrected:
        logger.info(
            "[VISION_HEADING] PyMuPDF anchoring corrected %d/%d heading(s)",
            corrected,
            len(merged),
        )
        merged.sort(key=lambda x: (x["page_start"], _norm(x["heading"])))
        for i, section in enumerate(merged):
            if i + 1 < len(merged):
                section["page_end"] = merged[i + 1]["page_start"]
            else:
                section["page_end"] = total_pages

    return merged


# ---------------------------------------------------------------------------
# Top-level scheme filtering
# ---------------------------------------------------------------------------

def _apply_top_level_scheme_filter(
    merged: List[Dict[str, Any]],
    total_pages: int,
) -> List[Dict[str, Any]]:
    """
    Keep only headings that fit the dominant top-level section scheme.

    This is the main fix for cases like:
      - true main sections are Roman-numbered
      - subsection-like headings such as "3. Post Training Phase" are mistakenly kept

    Rules:
    - always keep mandatory unnumbered headings
    - always drop obvious subsection-like headings
    - if dominant scheme is roman, strongly prefer roman + mandatory unnumbered + appendix
    - if dominant scheme is arabic, prefer arabic + mandatory unnumbered + appendix
    - if no dominant scheme, keep a broader set and let later scoring decide
    """
    if not merged:
        return []

    dominant = _dominant_top_level_scheme(merged)
    logger.info("[VISION_HEADING] Dominant top-level scheme: %s", dominant)

    filtered: List[Dict[str, Any]] = []

    for sec in merged:
        heading = sec["heading"]
        kind = _heading_prefix_kind(heading)
        stripped_norm = _norm(_strip_prefix(heading))

        if _is_subsection_like_heading(heading):
            logger.info("[VISION_HEADING] Scheme filter dropped subsection-like heading: %s", heading)
            continue

        if kind in {"mandatory-unnumbered", "appendix"}:
            filtered.append(sec)
            continue

        if dominant == "roman":
            if kind == "roman":
                filtered.append(sec)
            elif kind == "unnumbered" and stripped_norm in _COMMON_SECTION_WORDS:
                filtered.append(sec)
            else:
                logger.info("[VISION_HEADING] Scheme filter dropped non-roman heading: %s", heading)

        elif dominant == "arabic":
            if kind == "arabic":
                filtered.append(sec)
            elif kind == "unnumbered" and stripped_norm in _COMMON_SECTION_WORDS:
                filtered.append(sec)
            else:
                logger.info("[VISION_HEADING] Scheme filter dropped non-arabic heading: %s", heading)

        else:
            # Fallback when scheme is unclear
            if kind in {"roman", "arabic", "unnumbered"}:
                filtered.append(sec)

    if not filtered:
        return []

    filtered.sort(key=lambda x: (x["page_start"], _norm(x["heading"])))
    deduped: List[Dict[str, Any]] = []
    seen = set()
    for sec in filtered:
        key = (_norm(_strip_prefix(sec["heading"])), sec["page_start"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(sec)

    for i, section in enumerate(deduped):
        if i + 1 < len(deduped):
            section["page_end"] = deduped[i + 1]["page_start"]
        else:
            section["page_end"] = total_pages

    return deduped


# ---------------------------------------------------------------------------
# Heading confidence scoring
# ---------------------------------------------------------------------------

def _score_headings(
    merged: List[Dict[str, Any]],
    doc,
    total_pages: int,
    threshold: float = 0.62,
) -> List[Dict[str, Any]]:
    """
    Compute confidence score for each detected heading and remove low-confidence headings.

    Revised scoring:
    - separates "heading-like" from "top-level-heading-like"
    - penalizes subsection-style numbering when paper scheme suggests otherwise
    - rewards dominant scheme consistency
    - rewards page-top placement and stronger typography
    """
    if not merged:
        return []

    dominant_scheme = _dominant_top_level_scheme(merged)

    def _modal_font_size(page_dict: dict) -> float:
        sizes = []
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
        bbox = line.get("bbox")
        if not bbox or page_height <= 0:
            return False
        mid_y = (bbox[1] + bbox[3]) / 2.0
        return mid_y < page_height * 0.08 or mid_y > page_height * 0.92

    filtered: List[Dict[str, Any]] = []

    for sec in merged:
        heading = sec["heading"]
        page = sec["page_start"]
        kind = _heading_prefix_kind(heading)
        stripped_norm = _norm(_strip_prefix(heading))

        try:
            page_obj = doc[page - 1]
            page_dict = page_obj.get_text("dict", flags=0)
            page_height = page_obj.rect.height
        except Exception:
            filtered.append(sec)
            continue

        body_size = _modal_font_size(page_dict)

        # Base prior: GPT already proposed it
        gpt_score = 1.0

        text_match = 0.0
        typo_score = 0.0
        position_score = 0.0
        scheme_score = 0.5
        semantic_score = 0.0

        target = stripped_norm
        found = False

        for block in page_dict.get("blocks", []):
            if block.get("type") != 0:
                continue

            for line in block.get("lines", []):
                if _in_header_footer_zone(line, page_height):
                    continue

                spans = line.get("spans", [])
                line_text = "".join(s.get("text", "") for s in spans).strip()
                if not line_text:
                    continue

                line_norm = _norm(_strip_prefix(line_text))

                if target and target in line_norm:
                    text_match = 1.0

                    max_size = max(
                        (s.get("size", 0) for s in spans if s.get("text", "").strip()),
                        default=0,
                    )

                    if max_size >= body_size * 1.15:
                        typo_score = 1.0
                    elif max_size >= body_size * 1.05:
                        typo_score = 0.8
                    elif max_size >= body_size * 0.95:
                        typo_score = 0.55
                    else:
                        typo_score = 0.2

                    bbox = line.get("bbox")
                    if bbox and page_height > 0:
                        mid_y = (bbox[1] + bbox[3]) / 2.0
                        rel_y = mid_y / page_height
                        if rel_y <= 0.22:
                            position_score = 1.0
                        elif rel_y <= 0.35:
                            position_score = 0.7
                        elif rel_y <= 0.55:
                            position_score = 0.4
                        else:
                            position_score = 0.15

                    found = True
                    break

            if found:
                break

        # Scheme consistency
        if kind in {"mandatory-unnumbered", "appendix"}:
            scheme_score = 1.0
        elif dominant_scheme == "roman":
            if kind == "roman":
                scheme_score = 1.0
            elif kind == "arabic":
                scheme_score = 0.1
            elif kind == "unnumbered" and stripped_norm in _COMMON_SECTION_WORDS:
                scheme_score = 0.8
            else:
                scheme_score = 0.25
        elif dominant_scheme == "arabic":
            if kind == "arabic":
                scheme_score = 1.0
            elif kind == "roman":
                scheme_score = 0.2
            elif kind == "unnumbered" and stripped_norm in _COMMON_SECTION_WORDS:
                scheme_score = 0.8
            else:
                scheme_score = 0.3
        else:
            if kind in {"roman", "arabic"}:
                scheme_score = 0.8
            elif kind == "unnumbered" and stripped_norm in _COMMON_SECTION_WORDS:
                scheme_score = 0.8
            else:
                scheme_score = 0.45

        # Strong penalties for subsection-like forms
        if kind in {"decimal-subsection", "roman-hyphen-subsection", "alpha-subsection"}:
            scheme_score = 0.0

        # Semantic hint
        if stripped_norm in _COMMON_SECTION_WORDS or stripped_norm in _MANDATORY_UNNUMBERED_HEADINGS:
            semantic_score = 1.0
        elif any(word in stripped_norm for word in ["introduction", "method", "result", "discussion", "conclusion", "reference"]):
            semantic_score = 0.7
        else:
            semantic_score = 0.3

        confidence = (
            0.20 * gpt_score
            + 0.22 * text_match
            + 0.20 * typo_score
            + 0.14 * position_score
            + 0.16 * scheme_score
            + 0.08 * semantic_score
        )

        logger.info(
            "[VISION_HEADING] Confidence '%s' page %d → %.2f | kind=%s scheme=%s text=%.2f typo=%.2f pos=%.2f sem=%.2f",
            heading,
            page,
            confidence,
            kind,
            dominant_scheme,
            text_match,
            typo_score,
            position_score,
            semantic_score,
        )

        keep = confidence >= threshold

        # Extra hard guard for papers whose top-level style is Roman:
        # drop simple Arabic headings unless they are known mandatory headings.
        if dominant_scheme == "roman" and kind == "arabic" and stripped_norm not in _MANDATORY_UNNUMBERED_HEADINGS:
            logger.warning(
                "[VISION_HEADING] Dropped Arabic heading under Roman-dominant scheme: '%s' (%.2f)",
                heading,
                confidence,
            )
            keep = False

        if keep:
            filtered.append(sec)
        else:
            logger.warning(
                "[VISION_HEADING] Dropped low-confidence/non-top-level heading '%s' (%.2f)",
                heading,
                confidence,
            )

    if not filtered:
        return []

    filtered.sort(key=lambda x: (x["page_start"], _norm(x["heading"])))

    for i, section in enumerate(filtered):
        if i + 1 < len(filtered):
            section["page_end"] = filtered[i + 1]["page_start"]
        else:
            section["page_end"] = total_pages

    return filtered


# ---------------------------------------------------------------------------
# Infer missing numbered sections (gap detection)
# ---------------------------------------------------------------------------

def _infer_missing_numbered_sections(
    merged: List[Dict[str, Any]],
    doc,
    total_pages: int,
) -> List[Dict[str, Any]]:
    """
    Detect gaps in Roman / Arabic heading sequence and recover them generically.

    Recovery order:
    1. Try exact prefix-based recovery (e.g. "I.", "3.")
    2. If that fails, try generic heading-like line recovery in the plausible page window
       and synthesize the expected prefix onto the recovered heading text.

    This keeps recovery generic and avoids hardcoding "Introduction".
    """

    def _modal_font_size(page_dict: dict) -> float:
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
        bbox = line.get("bbox")
        if not bbox or page_height <= 0:
            return False
        mid_y = (bbox[1] + bbox[3]) / 2.0
        return mid_y < page_height * 0.08 or mid_y > page_height * 0.92

    def _line_looks_like_top_heading(
        line_text: str,
        spans: List[dict],
        body_size: float,
    ) -> bool:
        txt = line_text.strip()
        if not txt:
            return False

        txt_norm = _norm(txt)
        words = txt_norm.split()
        if not words:
            return False

        if len(txt_norm) > 120:
            return False
        if len(words) > 18:
            return False
        if txt.endswith("."):
            return False
        if re.match(r"^(figure|fig\.|table)\s+\d+", txt_norm):
            return False
        if re.match(r"^[A-Z]\.\s*$", txt.strip()):
            return False
        if re.match(r"^\d+\.\d+", txt.strip()):
            return False
        if re.match(r"^[A-Z]\.", txt.strip()) and len(words) <= 3:
            return False

        max_span_size = max(
            (s.get("size", 0) for s in spans if s.get("text", "").strip()),
            default=0,
        )
        if max_span_size < body_size * 0.98:
            return False

        return True

    def _search_pages_for_prefix(
        prefix_str: str,
        page_lo: int,
        page_hi: int,
    ) -> Optional[Tuple[int, str]]:
        prefix_norm = _norm(prefix_str)

        for pnum in range(page_lo, page_hi + 1):
            try:
                page = doc[pnum - 1]
                page_dict = page.get_text("dict", flags=0)
                body_size = _modal_font_size(page_dict)
                page_height = page.rect.height

                for block in page_dict.get("blocks", []):
                    if block.get("type") != 0:
                        continue

                    for line in block.get("lines", []):
                        if _in_header_footer_zone(line, page_height):
                            continue

                        spans = line.get("spans", [])
                        line_text = "".join(s.get("text", "") for s in spans).strip()
                        line_norm = _norm(line_text)

                        if not line_norm.startswith(prefix_norm):
                            continue

                        if not _line_looks_like_top_heading(line_text, spans, body_size):
                            continue

                        logger.info(
                            "[VISION_HEADING] Gap-recovery prefix-match: found '%s' on page %d",
                            line_text,
                            pnum,
                        )
                        return pnum, line_text

            except Exception as exc:
                logger.debug("[VISION_HEADING] Gap-recovery prefix search error page %d: %s", pnum, exc)

        return None

    def _search_pages_for_generic_heading_candidate(
        expected_prefix: str,
        page_lo: int,
        page_hi: int,
    ) -> Optional[Tuple[int, str]]:
        """
        Generic fallback:
        find a heading-like standalone line in the plausible page window,
        excluding lines that already belong to numbered headings we already know.
        """
        known_heading_norms = {
            _norm(_strip_prefix(sec["heading"])) for sec in merged
        }

        candidate_hits: List[Tuple[int, str, float]] = []

        for pnum in range(page_lo, page_hi + 1):
            try:
                page = doc[pnum - 1]
                page_dict = page.get_text("dict", flags=0)
                body_size = _modal_font_size(page_dict)
                page_height = page.rect.height

                for block in page_dict.get("blocks", []):
                    if block.get("type") != 0:
                        continue

                    for line in block.get("lines", []):
                        if _in_header_footer_zone(line, page_height):
                            continue

                        spans = line.get("spans", [])
                        line_text = "".join(s.get("text", "") for s in spans).strip()
                        if not line_text:
                            continue

                        line_norm = _norm(line_text)
                        stripped_norm = _norm(_strip_prefix(line_text))

                        if not _line_looks_like_top_heading(line_text, spans, body_size):
                            continue

                        if stripped_norm in known_heading_norms:
                            continue

                        if re.match(r"^[IVXivx]+[.)]\s+", line_text) or re.match(r"^\d+[.)]\s+", line_text):
                            continue

                        words = stripped_norm.split()
                        if len(words) < 1 or len(words) > 12:
                            continue

                        max_span_size = max(
                            (s.get("size", 0) for s in spans if s.get("text", "").strip()),
                            default=0,
                        )
                        score = float(max_span_size) - (0.01 * len(stripped_norm))
                        candidate_hits.append((pnum, line_text, score))

            except Exception as exc:
                logger.debug("[VISION_HEADING] Gap-recovery generic search error page %d: %s", pnum, exc)

        if not candidate_hits:
            return None

        candidate_hits.sort(key=lambda x: (x[0], -x[2]))
        best_page, best_text, _ = candidate_hits[0]
        synthesized = f"{expected_prefix} {best_text}".strip()

        logger.info(
            "[VISION_HEADING] Gap-recovery generic-match: synthesized '%s' on page %d from line '%s'",
            synthesized,
            best_page,
            best_text,
        )
        return best_page, synthesized

    numbered: List[Tuple[int, str, int]] = []
    for i, sec in enumerate(merged):
        h = sec["heading"].strip()
        m_r = _ROMAN_PREFIX_RE.match(h)
        m_a = _ARABIC_PREFIX_RE.match(h)
        if m_r:
            roman_str = m_r.group(1).upper()
            val = _ROMAN_TO_INT.get(roman_str)
            if val is not None:
                numbered.append((i, "roman", val))
        elif m_a:
            numbered.append((i, "arabic", int(m_a.group(1))))

    if not numbered:
        return merged

    kind_counts = Counter(k for _, k, _ in numbered)
    dominant_kind = kind_counts.most_common(1)[0][0]
    numbered = [(i, k, n) for i, k, n in numbered if k == dominant_kind]
    if not numbered:
        return merged

    detected_nums = {n for _, _, n in numbered}
    last_num = max(detected_nums)
    all_expected = set(range(1, last_num + 1))
    missing_nums = sorted(all_expected - detected_nums)

    if not missing_nums:
        return merged

    logger.info(
        "[VISION_HEADING] Gap detection: detected=%s  missing=%s  kind=%s",
        sorted(detected_nums),
        missing_nums,
        dominant_kind,
    )

    num_to_page: Dict[int, int] = {}
    for i, _, n in numbered:
        num_to_page[n] = merged[i]["page_start"]

    all_heading_pages = sorted(sec["page_start"] for sec in merged)

    new_sections: List[Dict[str, Any]] = []

    for miss in missing_nums:
        next_page = num_to_page.get(miss + 1, total_pages)

        prev_numbered_page = num_to_page.get(miss - 1)
        if prev_numbered_page is not None:
            search_lo = max(1, prev_numbered_page)
        else:
            preceding = [p for p in all_heading_pages if p < next_page]
            search_lo = max(preceding) if preceding else 1

        search_hi = min(total_pages, next_page)

        if dominant_kind == "roman":
            roman_str = _INT_TO_ROMAN.get(miss)
            if roman_str is None:
                continue
            prefix_str = roman_str + "."
        else:
            prefix_str = str(miss) + "."

        result = _search_pages_for_prefix(prefix_str, search_lo, search_hi)

        if result is None:
            result = _search_pages_for_generic_heading_candidate(
                expected_prefix=prefix_str,
                page_lo=search_lo,
                page_hi=search_hi,
            )

        if result is None:
            logger.info(
                "[VISION_HEADING] Gap-recovery: '%s' not recovered in pages %d-%d — skipping",
                prefix_str,
                search_lo,
                search_hi,
            )
            continue

        found_page, found_text = result
        new_sections.append(
            {
                "heading": found_text,
                "page_start": found_page,
                "page_end": -1,
                "_recovered": True,
            }
        )

    mandatory_unnumbered = ["abstract", "acknowledgements", "acknowledgments"]

    detected_normalised = {
        re.sub(r"^[IVXivx]+[.)]\s*|^\d+[.)]\s*", "", sec["heading"].lower()).strip()
        for sec in merged
    }

    for label in mandatory_unnumbered:
        if label in detected_normalised:
            continue

        if label == "abstract":
            search_lo, search_hi = 1, min(5, total_pages)
        else:
            search_lo, search_hi = max(1, total_pages - 10), total_pages

        result = _search_pages_for_prefix(label, search_lo, search_hi)
        if result is not None:
            found_page, found_text = result
            new_sections.append(
                {
                    "heading": found_text,
                    "page_start": found_page,
                    "page_end": -1,
                    "_recovered": True,
                }
            )
            logger.info(
                "[VISION_HEADING] Mandatory-heading recovery: inserted '%s' on page %d",
                found_text,
                found_page,
            )

    if not new_sections:
        return merged

    combined = merged + new_sections
    combined.sort(key=lambda x: (x["page_start"], _norm(x["heading"])))

    for i, sec in enumerate(combined):
        if i + 1 < len(combined):
            sec["page_end"] = combined[i + 1]["page_start"]
        else:
            sec["page_end"] = total_pages
        sec.pop("_recovered", None)

    logger.info(
        "[VISION_HEADING] Gap-recovery inserted %d missing section(s): %s",
        len(new_sections),
        [(s["heading"], s["page_start"]) for s in new_sections],
    )

    return combined


# ---------------------------------------------------------------------------
# Build SectionAssembly + HeadingTree from merged headings + PyMuPDF page text
# ---------------------------------------------------------------------------

def _build_sections_from_headings(
    merged_headings: List[Dict[str, Any]],
    page_texts: Dict[int, str],
    total_pages: int,
) -> Tuple["SectionAssembly", "HeadingTree"]:
    from pdf_ingestion.app.section_assembler import SectionAssembly, PaperSection
    from pdf_ingestion.app.structural_extractor import HeadingTree, Heading

    sections: List[PaperSection] = []
    headings: List[Heading] = []

    for idx, sec in enumerate(merged_headings):
        heading_text = sec["heading"]
        page_start = sec["page_start"]
        page_end = sec["page_end"]

        fetch_from = max(1, page_start - 1)

        text_parts: List[str] = []
        for pnum in range(fetch_from, page_end + 1):
            raw_page_text = (page_texts.get(pnum) or "").strip()
            if raw_page_text:
                text_parts.append(raw_page_text)

        content_text = "\n\n".join(text_parts).strip()

        sections.append(
            PaperSection(
                section_index=idx,
                heading_level=1,
                heading_text=heading_text,
                parent_heading=None,
                page_start=page_start,
                page_end=page_end,
                content_text=content_text,
                content_length=len(content_text),
            )
        )

        headings.append(
            Heading(
                level=1,
                text=heading_text,
                page=page_start,
            )
        )

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

    def _detect_and_extract(page) -> Tuple[str, bool]:
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

        full_blks = [b for b in blocks if (b["bbox"][2] - b["bbox"][0]) > pw * 0.80]
        full_ids = {id(b) for b in full_blks}
        non_full = [b for b in blocks if id(b) not in full_ids]

        is_two_col = False
        divider = pw / 2.0

        if len(non_full) >= 4:
            x0s = sorted({round(b["bbox"][0]) for b in non_full})
            if len(x0s) >= 2:
                gap_pairs = [
                    (x0s[i + 1] - x0s[i], (x0s[i] + x0s[i + 1]) / 2.0)
                    for i in range(len(x0s) - 1)
                ]
                best_gap, best_mid = max(gap_pairs, key=lambda g: g[0])

                if best_gap > pw * 0.20 and pw * 0.25 <= best_mid <= pw * 0.75:
                    left_blks = [b for b in non_full if b["bbox"][0] < best_mid]
                    right_blks = [b for b in non_full if b["bbox"][0] >= best_mid]
                    if len(left_blks) >= 2 and len(right_blks) >= 2:
                        is_two_col = True
                        divider = best_mid

        if is_two_col:
            col_left = sorted(
                [b for b in non_full if b["bbox"][0] < divider],
                key=lambda b: b["bbox"][1],
            )
            col_right = sorted(
                [b for b in non_full if b["bbox"][0] >= divider],
                key=lambda b: b["bbox"][1],
            )
            full_blks.sort(key=lambda b: b["bbox"][1])
            ordered = full_blks + col_left + col_right
        else:
            ordered = sorted(blocks, key=lambda b: (b["bbox"][1], b["bbox"][0]))

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
                page_texts[page_idx + 1] = text
            if is_two_col:
                two_col_pages += 1
        except Exception as e:
            logger.warning("[VISION_HEADING] Page %d text extraction failed: %s", page_idx + 1, e)

    if not page_texts:
        doc.close()
        return _err("PyMuPDF found no embedded text — PDF may be scanned.")

    layout_guess = "two-column" if two_col_pages > total_pages * 0.3 else "single-column"
    logger.info(
        "[VISION_HEADING] PyMuPDF extracted text from %d/%d pages | layout=%s (%d/%d pages two-column)",
        len(page_texts),
        total_pages,
        layout_guess,
        two_col_pages,
        total_pages,
    )

    batches: List[List[int]] = []
    start = 1
    while start <= total_pages:
        end = min(start + BATCH_SIZE - 1, total_pages)
        batches.append(list(range(start, end + 1)))
        if end == total_pages:
            break
        start = end - BATCH_OVERLAP + 1

    logger.info(
        "[VISION_HEADING] Sending %d batches to GPT-4o Vision (batch_size=%d, overlap=%d)",
        len(batches),
        BATCH_SIZE,
        BATCH_OVERLAP,
    )

    import concurrent.futures

    batch_results: List[List[Dict[str, Any]]] = [[] for _ in batches]

    def _run_batch(batch_idx: int, page_numbers: List[int]) -> List[Dict[str, Any]]:
        page_indices = [p - 1 for p in page_numbers]

        is_first_batch = 1 in page_numbers
        dpi = FIRST_BATCH_IMAGE_DPI if is_first_batch else IMAGE_DPI
        detail = FIRST_BATCH_IMAGE_DETAIL if is_first_batch else IMAGE_DETAIL
        tag = f"{page_numbers[0]}-{page_numbers[-1]}"

        return _call_vision_batch(
            llm_client=llm_client,
            doc=doc,
            page_indices=page_indices,
            page_numbers=page_numbers,
            dpi=dpi,
            detail=detail,
            batch_tag=tag,
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

    early_pages_results: List[Dict[str, Any]] = []
    early_pages = list(range(1, min(EARLY_PAGES_FOCUSED_COUNT, total_pages) + 1))
    if early_pages:
        logger.info(
            "[VISION_HEADING] Running focused early-pages pass on pages %s (dpi=%d, detail=%s)",
            early_pages,
            EARLY_PAGES_IMAGE_DPI,
            EARLY_PAGES_IMAGE_DETAIL,
        )
        early_pages_results = _call_vision_batch(
            llm_client=llm_client,
            doc=doc,
            page_indices=[p - 1 for p in early_pages],
            page_numbers=early_pages,
            dpi=EARLY_PAGES_IMAGE_DPI,
            detail=EARLY_PAGES_IMAGE_DETAIL,
            batch_tag=f"focused-{early_pages[0]}-{early_pages[-1]}",
        )

    all_results = batch_results[:]
    if early_pages_results:
        all_results.append(early_pages_results)

    merged = _merge_batch_results(all_results, total_pages)

    if merged:
        merged = _anchor_headings_with_pymupdf(merged, doc, total_pages)

    if merged:
        merged = _infer_missing_numbered_sections(merged, doc, total_pages)

    if merged:
        merged = _apply_top_level_scheme_filter(merged, total_pages)

    if merged:
        merged = _score_headings(merged, doc, total_pages)

    doc.close()

    if not merged:
        return _err(
            "GPT-4o Vision found no reliable top-level headings in the PDF. "
            "The paper may use an unusual layout."
        )

    logger.info(
        "[VISION_HEADING] Final merged headings (%d): %s",
        len(merged),
        [(m["heading"], m["page_start"], m["page_end"]) for m in merged],
    )

    assembly, heading_tree = _build_sections_from_headings(
        merged_headings=merged,
        page_texts=page_texts,
        total_pages=total_pages,
    )

    if assembly.is_empty():
        return _err("Sections were detected but all had empty content after text assembly.")

    logger.info(
        "[VISION_HEADING] Complete: %d sections, %d headings",
        len(assembly.sections),
        len(heading_tree.headings),
    )

    return assembly, heading_tree, total_pages