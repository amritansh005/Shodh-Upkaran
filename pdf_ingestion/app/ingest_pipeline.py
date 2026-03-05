"""
ingest_pipeline.py — Full pipeline: download → extract → chunk → embed → store.

Returns a structured result dict that routes directly to user-facing messages.
No fallbacks — failures return specific human-readable error reasons.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from pdf_ingestion.app.pdf_downloader import download_pdf
from pdf_ingestion.app.pdf_extractor import extract_pdf
from pdf_ingestion.app.chunker import chunk_pages
from pdf_ingestion.app.embedder import get_embedder
from pdf_ingestion.app.paper_store import PaperStore, STATUS_READY, STATUS_PROCESSING
from pdf_ingestion.app.structural_extractor import extract_headings

logger = logging.getLogger(__name__)


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

        # ── 3b. Extract headings via GPT-4o vision ───────────────────────────
        # Renders each page as an image and asks GPT-4o to identify headings
        # visually — same approach Claude uses when reading PDFs natively.
        # Stored in paper_headings column so heading queries skip embedding
        # retrieval entirely and answer directly from the stored structure.
        logger.info("[INGEST] Extracting headings via GPT-4o vision: %s", arxiv_id)
        try:
            heading_tree = extract_headings(
                pdf_bytes   = dl.pdf_bytes,
                endpoint    = getattr(settings, "azure_openai_endpoint",    "") if settings else "",
                api_key     = getattr(settings, "azure_openai_api_key",     "") if settings else "",
                api_version = getattr(settings, "azure_openai_api_version", "") if settings else "",
                deployment  = getattr(settings, "azure_openai_deployment",  "") if settings else "",
            )
            if not heading_tree.is_empty():
                store.save_headings(arxiv_id, heading_tree.to_json())
                logger.info(
                    "[INGEST] Headings stored: %s  count=%d",
                    arxiv_id, len(heading_tree.headings),
                )
            elif heading_tree.error:
                logger.warning(
                    "[INGEST] Heading extraction skipped: %s  reason=%s",
                    arxiv_id, heading_tree.error,
                )
            else:
                logger.info("[INGEST] No headings detected in: %s", arxiv_id)
        except Exception as _he:
            # Non-fatal: text extraction + chunking continues regardless
            logger.warning("[INGEST] Heading extraction raised unexpectedly: %s  err=%s", arxiv_id, _he)

        # ── 4. Extract text ───────────────────────────────────────────────────
        logger.info("[INGEST] Extracting text: %s", arxiv_id)
        extraction = extract_pdf(dl.pdf_bytes)

        if extraction.error:
            store.mark_failed(arxiv_id, extraction.error)
            return _fail(arxiv_id, f"Could not extract text from the PDF: {extraction.error}")

        if not extraction.pages:
            msg = "The PDF appears to contain no extractable text (may be fully image-based with no OCR support)."
            store.mark_failed(arxiv_id, msg)
            return _fail(arxiv_id, msg)

        logger.info(
            "[INGEST] Extracted: %s  pages=%d  used_ocr=%s",
            arxiv_id, extraction.total_pages, extraction.used_ocr,
        )

        # Log warning if any pages permanently failed
        if extraction.failed_pages_note:
            logger.warning("[INGEST] %s: %s", arxiv_id, extraction.failed_pages_note)

        # ── 5. Chunk ──────────────────────────────────────────────────────────
        chunks = chunk_pages(extraction.pages)
        if not chunks:
            msg = "The PDF was extracted but produced no text chunks. The paper may be empty or unreadable."
            store.mark_failed(arxiv_id, msg)
            return _fail(arxiv_id, msg)

        logger.info("[INGEST] Chunked: %s  chunks=%d", arxiv_id, len(chunks))

        # ── 6. Embed ──────────────────────────────────────────────────────────
        logger.info("[INGEST] Embedding: %s", arxiv_id)
        embedder = get_embedder()
        texts   = [c.text for c in chunks]
        vectors = embedder.embed_documents(texts)

        # ── 7. Store chunks ───────────────────────────────────────────────────
        store.delete_chunks(arxiv_id)
        chunk_rows = [
            (c.chunk_index, c.page_num, c.text, vectors[i])
            for i, c in enumerate(chunks)
        ]
        store.insert_chunks(arxiv_id, chunk_rows)

        try:
            store.ensure_vector_index()
        except Exception as ve:
            logger.warning("[INGEST] ensure_vector_index non-fatal: %s", ve)

        # ── 8. Mark ready ─────────────────────────────────────────────────────
        store.mark_ready(arxiv_id, extraction.total_pages, extraction.used_ocr)
        logger.info("[INGEST] Complete: %s  chunks=%d", arxiv_id, len(chunks))

        message = "Paper downloaded."
        if extraction.failed_pages_note:
            message += f"\n\n⚠️ Note: {extraction.failed_pages_note} Answers about content on those pages may be incomplete."

        return {
            "status":   "ready",
            "message":  message,
            "arxiv_id": arxiv_id,
            "chunks":   len(chunks),
            "used_ocr": extraction.used_ocr,
        }

    except Exception as e:
        error_msg = str(e)
        logger.error("[INGEST] Unexpected failure %s: %s", arxiv_id, error_msg, exc_info=True)
        try:
            store.mark_failed(arxiv_id, error_msg)
        except Exception:
            pass
        return _fail(arxiv_id, f"An unexpected error occurred during ingestion: {error_msg}")


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

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