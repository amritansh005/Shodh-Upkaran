"""
ingest_pipeline.py — Full pipeline: download → extract → chunk → embed → store.

Extraction strategy (in order, first success wins):
  1. PyMuPDF + GPT-4o Vision heading detection (fast, ~10-20s)
       - PyMuPDF extracts flat page text in ~1-3s
       - GPT-4o Vision detects top-level headings via parallel image batches
       - Page text is sliced by heading page ranges → PaperSection objects
       - Works for any born-digital PDF (the vast majority of arXiv papers)
       - Falls back to step 2 if: no embedded text (scanned) OR GPT-4o returns
         no headings OR LLM client is unavailable

  2. Marker GPU extraction (slow, ~10+ min, GPU required)
       - Full OCR + layout model pipeline
       - Handles scanned PDFs and complex layouts
       - Used only when step 1 cannot produce structured sections
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

        # ── 4. Build LLM client from settings ────────────────────────────────
        llm_client = _build_llm_client(settings)

        loop = asyncio.get_running_loop()

        assembly        = None
        heading_tree    = None
        pdf_total_pages = 0
        used_ocr        = False
        extraction_method = "unknown"

        # ── 5a. PRIMARY: PyMuPDF + GPT-4o Vision heading extraction ──────────
        if llm_client and llm_client.enabled():
            logger.info("[INGEST] Trying Vision heading extraction: %s", arxiv_id)
            try:
                from pdf_ingestion.app.vision_heading_extractor import (
                    extract_sections_via_vision_headings,
                )
                assembly, heading_tree, pdf_total_pages = await loop.run_in_executor(
                    _EXECUTOR,
                    lambda: extract_sections_via_vision_headings(
                        pdf_bytes  = dl.pdf_bytes,
                        llm_client = llm_client,
                        settings   = settings,
                    ),
                )

                if assembly and not assembly.is_empty():
                    extraction_method = "vision_heading"
                    logger.info(
                        "[INGEST] Vision heading extraction succeeded: %s  sections=%d",
                        arxiv_id, len(assembly.sections),
                    )
                else:
                    reason = getattr(assembly, "error", None) or "No sections produced."
                    logger.warning(
                        "[INGEST] Vision heading extraction failed for %s: %s "
                        "— falling back to Marker",
                        arxiv_id, reason,
                    )
                    assembly = None
                    heading_tree = None
                    pdf_total_pages = 0

            except Exception as e:
                logger.error(
                    "[INGEST] Vision heading extraction exception for %s: %s "
                    "— falling back to Marker",
                    arxiv_id, e, exc_info=True,
                )
                assembly = None
                heading_tree = None
                pdf_total_pages = 0
        else:
            logger.info(
                "[INGEST] LLM client not available — skipping Vision heading extraction, "
                "going straight to Marker: %s",
                arxiv_id,
            )

        # ── 5b. FALLBACK: Marker GPU extraction ───────────────────────────────
        if assembly is None or assembly.is_empty():
            logger.info("[INGEST] Starting Marker extraction: %s", arxiv_id)
            try:
                from pdf_ingestion.app.vision_extractor import extract_sections_via_vision

                assembly, heading_tree, pdf_total_pages = await loop.run_in_executor(
                    _EXECUTOR,
                    lambda: extract_sections_via_vision(
                        pdf_bytes = dl.pdf_bytes,
                        settings  = settings,
                    ),
                )
                extraction_method = "marker"
                used_ocr = True
            except Exception as e:
                error_msg = f"Marker extraction exception: {e}"
                logger.error("[INGEST] %s for %s", error_msg, arxiv_id, exc_info=True)
                store.mark_failed(arxiv_id, error_msg)
                return _fail(arxiv_id, error_msg)

        # ── 6. Validate extraction result ─────────────────────────────────────
        if assembly is None or assembly.is_empty():
            reason = getattr(assembly, "error", None) or "All extraction methods returned no sections."
            logger.error(
                "[INGEST] All extraction methods failed: %s  method=%s  reason=%s",
                arxiv_id, extraction_method, reason,
            )
            store.mark_failed(arxiv_id, reason)
            return _fail(arxiv_id, reason)

        logger.info(
            "[INGEST] Extraction complete: %s  method=%s  sections=%d  pages=%d",
            arxiv_id, extraction_method, len(assembly.sections), pdf_total_pages,
        )

        # ── 7. Store headings + sections ──────────────────────────────────────
        if heading_tree and not heading_tree.is_empty():
            store.save_headings(arxiv_id, heading_tree.to_json())
            logger.info(
                "[INGEST] Headings stored: %s  count=%d",
                arxiv_id, len(heading_tree.headings),
            )

        store.delete_sections(arxiv_id)
        store.insert_sections(arxiv_id, assembly.sections)
        logger.info(
            "[INGEST] Sections stored: %s  count=%d",
            arxiv_id, len(assembly.sections),
        )

        # ── 8. Chunk ──────────────────────────────────────────────────────────
        chunks = chunk_sections(assembly.sections)

        if not chunks:
            msg = "Extraction produced sections but chunker returned no text chunks."
            store.mark_failed(arxiv_id, msg)
            return _fail(arxiv_id, msg)

        logger.info("[INGEST] Chunked: %s  chunks=%d", arxiv_id, len(chunks))

        # ── 9. Embed ──────────────────────────────────────────────────────────
        logger.info("[INGEST] Embedding: %s", arxiv_id)
        embedder = get_embedder()
        vectors  = embedder.embed_documents([c.text for c in chunks])

        # ── 10. Store chunks ──────────────────────────────────────────────────
        store.delete_chunks(arxiv_id)
        store.insert_chunks(arxiv_id, [
            (c.chunk_index, c.page_num, c.text, vectors[i], c.section_heading)
            for i, c in enumerate(chunks)
        ])

        try:
            store.ensure_vector_index()
        except Exception as ve:
            logger.warning("[INGEST] ensure_vector_index non-fatal: %s", ve)

        # ── 11. Mark ready ────────────────────────────────────────────────────
        store.mark_ready(arxiv_id, pdf_total_pages, used_ocr)
        logger.info(
            "[INGEST] Complete: %s  method=%s  chunks=%d  pages=%d",
            arxiv_id, extraction_method, len(chunks), pdf_total_pages,
        )

        return {
            "status":   "ready",
            "message":  "Paper downloaded.",
            "arxiv_id": arxiv_id,
            "chunks":   len(chunks),
            "used_ocr": used_ocr,
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

def _build_llm_client(settings):
    """
    Build an LLMClient from ingest settings (sourced from app.state.settings in main.py).
    Returns None if settings are unavailable or any credential is missing.
    """
    if settings is None:
        return None
    try:
        from pdf_ingestion.app.llm_client import LLMClient
        return LLMClient(
            endpoint    = getattr(settings, "azure_openai_endpoint",    "") or "",
            api_key     = getattr(settings, "azure_openai_api_key",     "") or "",
            api_version = getattr(settings, "azure_openai_api_version", "") or "",
            deployment  = getattr(settings, "azure_openai_deployment",  "") or "",
        )
    except Exception as e:
        logger.warning("[INGEST] Could not build LLM client from settings: %s", e)
        return None


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