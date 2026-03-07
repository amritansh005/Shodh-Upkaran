"""
ingest_pipeline.py — Full pipeline: download → Marker extract → chunk → embed → store.
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict

from pdf_ingestion.app.pdf_downloader import download_pdf
from pdf_ingestion.app.chunker import chunk_sections
from pdf_ingestion.app.embedder import get_embedder
from pdf_ingestion.app.paper_store import PaperStore, STATUS_READY, STATUS_PROCESSING
from pdf_ingestion.app.vision_extractor import extract_sections_via_vision

logger = logging.getLogger(__name__)

_EXECUTOR = ThreadPoolExecutor(max_workers=2)


async def ingest_paper(paper: Dict[str, Any], store: PaperStore, settings=None) -> Dict[str, Any]:
    arxiv_id = str(paper.get("arxiv_id") or paper.get("id") or "").strip()
    title    = str(paper.get("title") or "").strip()
    pdf_url  = _resolve_pdf_url(paper)

    if not arxiv_id:
        return _fail("", "Could not determine the arXiv ID for this paper.")

    if not pdf_url:
        return _fail(arxiv_id, "No PDF URL is available for this paper.")

    # ── 1. Already ingested? ──────────────────────────────────────────────────
    existing_status = store.get_paper_status(arxiv_id)

    if existing_status == STATUS_READY and store.chunk_count(arxiv_id) > 0:
        logger.info("[INGEST] Already ready: %s", arxiv_id)
        return {
            "status":   "already_ready",
            "message":  "Paper downloaded.",
            "arxiv_id": arxiv_id,
            "chunks":   store.chunk_count(arxiv_id),
            "used_ocr": False,
        }

    if existing_status == STATUS_PROCESSING:
        logger.warning("[INGEST] Found stuck 'processing' status for %s — re-ingesting", arxiv_id)

    # ── 2. Save metadata + mark processing ───────────────────────────────────
    authors_str    = _format_authors(paper)
    abstract       = str(paper.get("abstract") or paper.get("summary") or "").strip()
    published_date = str(paper.get("published") or paper.get("published_date") or "").strip()

    store.upsert_paper_meta(
        arxiv_id=arxiv_id,
        title=title,
        authors=authors_str,
        pdf_url=pdf_url,
        abstract=abstract,
        published_date=published_date,
        status=STATUS_PROCESSING,
    )

    try:
        # ── 3. Download PDF ───────────────────────────────────────────────────
        logger.info("[INGEST] Downloading: %s  url=%s", arxiv_id, pdf_url)
        dl = await download_pdf(pdf_url)

        if not dl.success:
            store.mark_failed(arxiv_id, dl.error_message or "Download failed.")
            return _fail(arxiv_id, dl.error_message)

        store.save_pdf_bytes(arxiv_id, dl.pdf_bytes)
        logger.info("[INGEST] Downloaded: %s  bytes=%d", arxiv_id, len(dl.pdf_bytes))

        # ── 4. Marker extraction ──────────────────────────────────────────────
        logger.info("[INGEST] Starting Marker extraction: %s", arxiv_id)

        loop = asyncio.get_running_loop()

        assembly, heading_tree, pdf_total_pages = await loop.run_in_executor(
            _EXECUTOR,
            lambda: extract_sections_via_vision(pdf_bytes=dl.pdf_bytes, settings=settings),
        )

        # ── 5. Check extraction result ────────────────────────────────────────
        if assembly is None or assembly.is_empty():
            reason = getattr(assembly, "error", None) or "Marker returned no sections."
            logger.error("[INGEST] Extraction failed: %s  reason=%s", arxiv_id, reason)
            store.mark_failed(arxiv_id, reason)
            return _fail(arxiv_id, reason)

        # ── 6. Store headings + sections ──────────────────────────────────────
        if not heading_tree.is_empty():
            store.save_headings(arxiv_id, heading_tree.to_json())
            logger.info("[INGEST] Headings stored: %s  count=%d", arxiv_id, len(heading_tree.headings))

        store.delete_sections(arxiv_id)
        store.insert_sections(arxiv_id, assembly.sections)
        logger.info("[INGEST] Sections stored: %s  count=%d", arxiv_id, len(assembly.sections))

        # ── 7. Chunk ──────────────────────────────────────────────────────────
        chunks = chunk_sections(assembly.sections)

        if not chunks:
            msg = "Marker extracted sections but produced no text chunks."
            store.mark_failed(arxiv_id, msg)
            return _fail(arxiv_id, msg)

        logger.info("[INGEST] Chunked: %s  chunks=%d", arxiv_id, len(chunks))

        # ── 8. Embed ──────────────────────────────────────────────────────────
        logger.info("[INGEST] Embedding: %s", arxiv_id)
        embedder = get_embedder()
        vectors  = embedder.embed_documents([c.text for c in chunks])

        # ── 9. Store chunks ───────────────────────────────────────────────────
        store.delete_chunks(arxiv_id)
        store.insert_chunks(arxiv_id, [
            (c.chunk_index, c.page_num, c.text, vectors[i], c.section_heading)
            for i, c in enumerate(chunks)
        ])

        try:
            store.ensure_vector_index()
        except Exception as ve:
            logger.warning("[INGEST] ensure_vector_index non-fatal: %s", ve)

        # ── 10. Mark ready ────────────────────────────────────────────────────
        store.mark_ready(arxiv_id, pdf_total_pages, False)
        logger.info("[INGEST] Complete: %s  chunks=%d  pages=%d", arxiv_id, len(chunks), pdf_total_pages)

        return {
            "status":   "ready",
            "message":  "Paper downloaded.",
            "arxiv_id": arxiv_id,
            "chunks":   len(chunks),
            "used_ocr": False,
        }

    except Exception as e:
        error_msg = str(e)
        logger.error("[INGEST] Unexpected failure %s: %s", arxiv_id, error_msg, exc_info=True)
        try:
            store.mark_failed(arxiv_id, error_msg)
        except Exception:
            pass
        return _fail(arxiv_id, f"An unexpected error occurred during ingestion: {error_msg}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fail(arxiv_id: str, message: str) -> Dict[str, Any]:
    return {
        "status":   "failed",
        "message":  message,
        "arxiv_id": arxiv_id,
        "chunks":   0,
        "used_ocr": False,
    }


def _resolve_pdf_url(paper: Dict[str, Any]) -> str:
    pdf = (
        paper.get("pdf_url")
        or paper.get("pdf")
        or paper.get("pdfLink")
        or paper.get("pdf_link")
        or ""
    )
    if isinstance(pdf, dict):
        pdf = pdf.get("href") or pdf.get("url") or ""
    pdf = str(pdf or "").strip()
    if pdf:
        return pdf
    arxiv_id = str(paper.get("arxiv_id") or paper.get("id") or "").strip()
    if arxiv_id:
        return f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    return ""


def _format_authors(paper: Dict[str, Any]) -> str:
    authors = paper.get("authors") or paper.get("author") or []
    if isinstance(authors, list):
        return ", ".join(str(a).strip() for a in authors if str(a).strip())
    return str(authors or "").strip()