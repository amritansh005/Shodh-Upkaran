"""
structural_extractor.py — Extract document headings from a PDF by rendering
each page as an image and sending it to GPT-4o vision (Azure OpenAI).

This is how vision models natively read PDFs: page images let the model detect
headings visually (bold text, font size, indentation) rather than relying on
text extraction that flattens document hierarchy.

The result is stored once at ingest time in the papers.paper_headings column.
Heading queries in qa_service then bypass embedding retrieval entirely.

Performance: all batches are dispatched CONCURRENTLY using ThreadPoolExecutor
so the 5 API calls for a 44-page paper run in parallel instead of sequentially.
Typical speedup: 5x (wall time ≈ single slowest batch instead of sum of all).
"""

from __future__ import annotations

import base64
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_DPI        = 150
_ZOOM       = _DPI / 72.0
_BATCH_SIZE = 10
_MAX_TOKENS = 2048

# Max parallel API calls. Azure OpenAI rate limits vary by deployment tier.
# 5 is safe for most standard deployments — lower to 3 if you see 429 errors.
_MAX_WORKERS = 5

_SYSTEM_PROMPT = """\
You are a document structure analyser. Your sole task is to extract the
section and sub-section headings from the page images of a research paper.

Rules:
- Return ONLY valid JSON — no markdown fences, no commentary, no explanation.
- The JSON must be a list of objects, each with exactly these keys:
    "level" : integer  1 = top-level section  (e.g. "I. Introduction")
                       2 = subsection          (e.g. "A. Background")
                       3 = sub-subsection      (e.g. "1. Named Entity Recognition")
    "text"  : string   The heading text EXACTLY as it appears on the page,
                       including any numbering (e.g. "III. Research Methodology")
    "page"  : integer  The 1-based page number where this heading appears
- Include  : section headings, subsection headings, sub-subsection headings.
- Exclude  : figure captions, table titles, author names, abstract label,
             keywords label, body paragraph text, page numbers, footnotes.
- If a page contains NO headings, contribute nothing for that page.
- Preserve exact capitalisation and numbering from the page.
- Output an empty list [] if no headings are found across all pages.
"""

_USER_PROMPT = (
    "These are page images from a research paper (pages {start}-{end}).\n"
    "Extract all section headings exactly as described in your instructions.\n"
    "Return ONLY the JSON list."
)


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class Heading:
    level: int
    text:  str
    page:  int


@dataclass
class HeadingTree:
    headings: List[Heading] = field(default_factory=list)
    error:    Optional[str] = None

    def to_json(self) -> str:
        return json.dumps(
            [{"level": h.level, "text": h.text, "page": h.page} for h in self.headings],
            ensure_ascii=False,
        )

    @classmethod
    def from_json(cls, raw: str) -> "HeadingTree":
        try:
            items = json.loads(raw or "[]")
            headings = [
                Heading(level=int(it["level"]), text=str(it["text"]).strip(), page=int(it["page"]))
                for it in items
                if isinstance(it, dict) and str(it.get("text", "")).strip()
            ]
            return cls(headings=headings)
        except Exception as exc:
            return cls(headings=[], error=f"parse error: {exc}")

    def is_empty(self) -> bool:
        return not self.headings

    def format_for_llm(self) -> str:
        lines = []
        for h in self.headings:
            indent = "  " * (h.level - 1)
            lines.append(f"{indent}- {h.text}  (page {h.page})")
        return "\n".join(lines)


# ── Azure OpenAI client ───────────────────────────────────────────────────────

def _build_azure_client(
    endpoint: str,
    api_key: str,
    api_version: str,
    deployment: str,
):
    """
    Build AzureOpenAI client from credentials passed in explicitly.
    Called from ingest_pipeline which already has the Settings object.
    Returns (client, deployment_name) or raises RuntimeError if not configured.
    """
    from openai import AzureOpenAI

    if not endpoint:
        raise RuntimeError("AZURE_OPENAI_ENDPOINT is not set in pdf_ingestion/.env")
    if not api_key:
        raise RuntimeError("AZURE_OPENAI_API_KEY is not set in pdf_ingestion/.env")

    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
    )
    return client, deployment


# ── Single batch call (runs in thread) ───────────────────────────────────────

def _process_batch(
    bidx: int,
    batch: List[Tuple[int, str]],
    client,
    deployment: str,
) -> Tuple[int, List[Heading]]:
    """
    Send one batch of page images to GPT-4o vision and parse the result.
    Returns (bidx, headings_list) so results can be reassembled in order.
    Runs inside a ThreadPoolExecutor thread — must not mutate shared state.
    """
    start_p, end_p = batch[0][0], batch[-1][0]
    logger.info(
        "[STRUCTURAL] Batch %d — pages %d-%d via Azure GPT-4o vision (parallel)",
        bidx + 1, start_p, end_p,
    )

    # Build message content
    content = []
    for page_num, b64 in batch:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{b64}",
                "detail": "high",
            },
        })
        content.append({"type": "text", "text": f"(Page {page_num})"})

    content.append({
        "type": "text",
        "text": _USER_PROMPT.format(start=start_p, end=end_p),
    })

    try:
        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": content},
            ],
            max_tokens=_MAX_TOKENS,
            temperature=0.0,
        )
        raw  = response.choices[0].message.content or ""
        text = raw.strip()

        # Strip markdown fences the model sometimes adds despite instructions
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        items = json.loads(text)
        headings = [
            Heading(
                level=int(it.get("level", 1)),
                text=str(it.get("text", "")).strip(),
                page=int(it.get("page", start_p)),
            )
            for it in items
            if isinstance(it, dict) and str(it.get("text", "")).strip()
        ]
        logger.info("[STRUCTURAL] Batch %d: %d headings found", bidx + 1, len(headings))
        return bidx, headings

    except json.JSONDecodeError as exc:
        logger.warning(
            "[STRUCTURAL] Batch %d JSON parse failed: %s — skipping batch",
            bidx + 1, exc,
        )
        return bidx, []

    except Exception as exc:
        logger.error(
            "[STRUCTURAL] Batch %d API call failed: %s — skipping batch",
            bidx + 1, exc, exc_info=True,
        )
        return bidx, []


# ── Main entry point ──────────────────────────────────────────────────────────

def extract_headings(
    pdf_bytes: bytes,
    endpoint: str,
    api_key: str,
    api_version: str,
    deployment: str,
) -> HeadingTree:
    """
    Render each PDF page as a PNG image and extract headings via GPT-4o vision.

    Credentials are passed in explicitly from ingest_pipeline, which gets them
    from the Settings object loaded by main.py — this avoids relying on
    os.environ which is not populated when pydantic-settings loads from a file.

    All batches are dispatched concurrently (ThreadPoolExecutor) so wall time
    equals the slowest single batch rather than the sum of all batches.
    For a 44-page paper (5 batches) this is typically a 4-5x speedup.

    - Never raises — returns HeadingTree with error set on failure
    """
    # ── Build Azure client ────────────────────────────────────────────────────
    try:
        client, deployment = _build_azure_client(endpoint, api_key, api_version, deployment)
    except RuntimeError as exc:
        logger.warning("[STRUCTURAL] %s — skipping heading extraction.", exc)
        return HeadingTree(error=str(exc))

    # ── Render PDF pages to PNG images ────────────────────────────────────────
    try:
        import fitz  # PyMuPDF — already in requirements.txt
    except ImportError:
        msg = "PyMuPDF (fitz) not installed — run: pip install pymupdf"
        logger.error("[STRUCTURAL] %s", msg)
        return HeadingTree(error=msg)

    try:
        doc     = fitz.open(stream=pdf_bytes, filetype="pdf")
        n_pages = len(doc)
        matrix  = fitz.Matrix(_ZOOM, _ZOOM)
        logger.info(
            "[STRUCTURAL] Rendering %d pages at %d DPI for heading extraction",
            n_pages, _DPI,
        )

        page_images: List[Tuple[int, str]] = []
        for i in range(n_pages):
            pix = doc[i].get_pixmap(matrix=matrix, colorspace=fitz.csRGB)
            b64 = base64.standard_b64encode(pix.tobytes("png")).decode("ascii")
            page_images.append((i + 1, b64))
        doc.close()

    except Exception as exc:
        msg = f"PDF rendering failed: {exc}"
        logger.error("[STRUCTURAL] %s", msg, exc_info=True)
        return HeadingTree(error=msg)

    # ── Dispatch all batches concurrently ─────────────────────────────────────
    batches = [
        page_images[i: i + _BATCH_SIZE]
        for i in range(0, len(page_images), _BATCH_SIZE)
    ]
    n_batches = len(batches)
    workers   = min(_MAX_WORKERS, n_batches)

    logger.info(
        "[STRUCTURAL] Dispatching %d batches concurrently (max_workers=%d)",
        n_batches, workers,
    )

    # results_map: bidx → List[Heading], preserves batch order after futures resolve
    results_map: Dict[int, List[Heading]] = {}

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_process_batch, bidx, batch, client, deployment): bidx
            for bidx, batch in enumerate(batches)
        }
        for future in as_completed(futures):
            try:
                bidx, headings = future.result()
                results_map[bidx] = headings
            except Exception as exc:
                bidx = futures[future]
                logger.error("[STRUCTURAL] Batch %d future raised: %s", bidx + 1, exc)
                results_map[bidx] = []

    # Reassemble in original batch order
    all_headings: List[Heading] = []
    for bidx in sorted(results_map):
        all_headings.extend(results_map[bidx])

    # ── Deduplicate and sort by document order ────────────────────────────────
    seen: set = set()
    unique: List[Heading] = []
    for h in sorted(all_headings, key=lambda x: (x.page, x.level)):
        key = (h.page, h.text.lower())
        if key not in seen:
            seen.add(key)
            unique.append(h)

    logger.info(
        "[STRUCTURAL] Complete — %d unique headings extracted from %d pages",
        len(unique), n_pages,
    )
    return HeadingTree(headings=unique)