"""
pdf_extractor.py — Docling PDF extraction with batched page processing and VRAM management.

Strategy:
- Split the PDF into page-range batches (default: 10 pages per batch).
- For each batch:
    Pass 1 — Docling StandardPipeline without OCR (full layout, tables, formulas).
    Pass 2 — Docling StandardPipeline with OCR    (only failed pages from Pass 1).
    Pass 3 — PyMuPDF rasterise → RapidOCR         (only failed pages from Pass 2).
    Pass 4 — Docling SimplePipeline               (only failed pages from Pass 3).
- Between every Docling pass: destroy the converter, flush VRAM and run GC.
  This prevents GPU memory accumulation across batches (RTX 2060 / 6 GB VRAM).
- PyMuPDF is used for: page counting, byte slicing, rasterisation (Pass 3), and
  text layer extraction via Docling SimplePipeline (Pass 4).

Docling 2.75.0 API used throughout.
"""

from __future__ import annotations

import gc
import logging
import os
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_PAGE_BREAK_TOKEN = "\n\n<<<PAGE_BREAK>>>\n\n"

# Number of PDF pages processed per Docling batch.
# Lower  → less VRAM per batch, more converter rebuilds.
# Higher → more VRAM per batch, fewer rebuilds (faster).
# 10 is safe for RTX 2060 (6 GB) with typical arxiv papers.
BATCH_SIZE: int = 10

# DPI for PyMuPDF rasterisation in Pass 3.
# 200 is a good balance — high enough for RapidOCR accuracy, low enough to be fast.
RASTER_DPI: int = 200


# ---------------------------------------------------------------------------
# Data model (compatible with existing pipeline)
# ---------------------------------------------------------------------------

@dataclass
class PageResult:
    page_num: int          # 1-based, relative to the FULL document
    text: str
    tables: List[str] = field(default_factory=list)
    is_ocr: bool = False


@dataclass
class ExtractionResult:
    pages: List[PageResult] = field(default_factory=list)
    total_pages: int = 0
    used_ocr: bool = False
    error: Optional[str] = None
    failed_pages_note: Optional[str] = None

    def full_text(self) -> str:
        parts: List[str] = []
        for p in self.pages:
            if p.text.strip():
                parts.append(p.text)
            for t in p.tables:
                if t.strip():
                    parts.append(t)
        return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# VRAM flush
# ---------------------------------------------------------------------------

def _flush_vram() -> None:
    """Release cached GPU memory and run Python GC."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass
    gc.collect()


# ---------------------------------------------------------------------------
# Docling converter factories
# ---------------------------------------------------------------------------

def _build_converter(use_ocr: bool = False):
    """
    Build a fresh Docling StandardPipeline converter.
    Full layout analysis, table detection, formula handling.
    use_ocr=True adds RapidOCR for scanned/image pages.
    Always del the returned object and call _flush_vram() afterwards.
    """
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, AcceleratorOptions
    from docling.datamodel.accelerator_options import AcceleratorDevice

    try:
        import torch
        cuda_available = torch.cuda.is_available()
    except ImportError:
        cuda_available = False

    device = AcceleratorDevice.CUDA if cuda_available else AcceleratorDevice.CPU

    if cuda_available:
        logger.info("[EXTRACT] CUDA available — using GPU for Docling.")
    else:
        logger.warning("[EXTRACT] CUDA not available — falling back to CPU.")

    accelerator_options = AcceleratorOptions(num_threads=1, device=device)
    pipeline_options = PdfPipelineOptions(
        accelerator_options=accelerator_options,
        do_ocr=use_ocr,
    )

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )


def _build_simple_converter():
    """
    Build a fresh Docling SimplePipeline converter.
    Skips layout/table/OCR models — uses Docling's internal PyMuPDF-backed
    fast text extraction. No GPU needed. Used as Pass 4 last resort.
    Always del the returned object and call _flush_vram() afterwards.
    """
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.pipeline.simple_pipeline import SimplePipeline

    pipeline_options = PdfPipelineOptions()

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=SimplePipeline,
                pipeline_options=pipeline_options,
            )
        }
    )


# ---------------------------------------------------------------------------
# PDF page slicer — PyMuPDF (bytes slicing only, NO text extraction)
# ---------------------------------------------------------------------------

def _slice_pdf_pages(pdf_bytes: bytes, start_0: int, end_0: int) -> Optional[bytes]:
    """
    Slice pages [start_0, end_0] (both 0-based inclusive) from pdf_bytes.
    Returns new PDF bytes or None on failure.
    """
    try:
        import fitz
        src = fitz.open(stream=pdf_bytes, filetype="pdf")
        dst = fitz.open()
        dst.insert_pdf(src, from_page=start_0, to_page=end_0)
        sliced = dst.write()
        src.close()
        dst.close()
        return sliced
    except Exception as e:
        logger.warning("[EXTRACT] PyMuPDF slice failed (pages %d-%d): %s", start_0, end_0, e)
        return None


def _get_total_pages(pdf_bytes: bytes) -> int:
    """Return total page count using PyMuPDF (fast, no GPU)."""
    try:
        import fitz
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        n = doc.page_count
        doc.close()
        return n
    except Exception as e:
        logger.warning("[EXTRACT] Could not count pages via PyMuPDF: %s", e)
        return 0


# ---------------------------------------------------------------------------
# Pass 3: PyMuPDF rasterise → RapidOCR (only for specific failed pages)
# ---------------------------------------------------------------------------

def _rasterise_and_ocr(
    batch_bytes: bytes,
    failed_local_indices: List[int],
    global_page_offset: int,
) -> List[PageResult]:
    """
    Pass 3: Rasterise only the specific failed pages using PyMuPDF at RASTER_DPI,
    then feed each page image directly into RapidOCR (forced torch engine).

    failed_local_indices: 0-based indices within the batch that still need recovery.
    global_page_offset:   0-based index of this batch's first page in the full doc.
    """
    try:
        import fitz
    except ImportError:
        logger.warning("[EXTRACT] PyMuPDF (fitz) not available for Pass 3 rasterisation.")
        return []

    try:
        from rapidocr import RapidOCR
        ocr = RapidOCR()
        logger.info("[EXTRACT] Pass 3: RapidOCR initialised.")
    except Exception as e:
        logger.warning("[EXTRACT] RapidOCR not available for Pass 3: %s", e)
        return []

    pages: List[PageResult] = []

    try:
        doc = fitz.open(stream=batch_bytes, filetype="pdf")

        for local_idx in failed_local_indices:
            global_page_num = global_page_offset + local_idx + 1  # 1-based

            if local_idx >= len(doc):
                logger.warning(
                    "[EXTRACT] Pass 3: local_idx %d out of range for batch with %d pages.",
                    local_idx, len(doc),
                )
                continue

            try:
                page = doc[local_idx]
                pix = page.get_pixmap(dpi=RASTER_DPI)
                image_bytes = pix.tobytes("png")

                result, _ = ocr(image_bytes)

                if result:
                    lines = [item[1] for item in result if item[1] and item[1].strip()]
                    text = "\n".join(lines).strip()
                    if text:
                        pages.append(PageResult(
                            page_num=global_page_num,
                            text=text,
                            tables=[],
                            is_ocr=True,
                        ))
                        logger.info(
                            "[EXTRACT] Pass 3: page %d recovered (%d chars).",
                            global_page_num, len(text),
                        )
                    else:
                        logger.info(
                            "[EXTRACT] Pass 3: page %d — OCR returned no text.",
                            global_page_num,
                        )
                else:
                    logger.info(
                        "[EXTRACT] Pass 3: page %d — no OCR result.",
                        global_page_num,
                    )

            except Exception as e:
                logger.warning(
                    "[EXTRACT] Pass 3 failed on page %d: %s",
                    global_page_num, e,
                )

        doc.close()

    except Exception as e:
        logger.warning("[EXTRACT] Pass 3 batch failed: %s", e)

    return pages


# ---------------------------------------------------------------------------
# Single-batch Docling run (shared by Pass 1, 2, and 4)
# ---------------------------------------------------------------------------

def _run_docling_on_bytes(
    pdf_bytes: bytes,
    global_page_offset: int,
    use_ocr: bool = False,
    simple: bool = False,
) -> Tuple[List[PageResult], List[int]]:
    """
    Run Docling on a pre-sliced PDF byte blob.

    Parameters
    ----------
    pdf_bytes           : bytes of the sliced batch PDF
    global_page_offset  : 0-based index of this batch's first page in the full doc
    use_ocr             : (StandardPipeline only) enable RapidOCR
    simple              : if True, use SimplePipeline instead of StandardPipeline

    Returns
    -------
    (pages, failed_local_indices)
        pages                : PageResult list with correct global page_num
        failed_local_indices : 0-based indices within this batch with no text
    """
    try:
        from docling_core.types.doc import ImageRefMode
    except Exception as e:
        logger.error("[EXTRACT] docling_core import failed: %s", e)
        return [], []

    converter = None
    markdown = None

    try:
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp_path = tmp.name
                tmp.write(pdf_bytes)
                tmp.flush()

            converter = _build_simple_converter() if simple else _build_converter(use_ocr=use_ocr)
            conv = converter.convert(tmp_path, raises_on_error=False)

            doc = getattr(conv, "document", None)
            if doc is None:
                logger.warning(
                    "[EXTRACT] Docling returned no document (simple=%s, use_ocr=%s)",
                    simple, use_ocr,
                )
                return [], []

            markdown = doc.export_to_markdown(
                page_break_placeholder=_PAGE_BREAK_TOKEN,
                image_mode=ImageRefMode.PLACEHOLDER,
            )
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    except Exception as e:
        logger.warning(
            "[EXTRACT] Docling batch failed (simple=%s, use_ocr=%s): %s",
            simple, use_ocr, e,
        )
        return [], []

    finally:
        if converter is not None:
            del converter
        _flush_vram()

    if not markdown or not markdown.strip():
        return [], []

    raw_pages = markdown.split(_PAGE_BREAK_TOKEN)
    pages: List[PageResult] = []
    failed_local_indices: List[int] = []

    for local_idx, page_text in enumerate(raw_pages):
        cleaned = (page_text or "").strip()
        global_page_num = global_page_offset + local_idx + 1  # 1-based

        if cleaned:
            pages.append(PageResult(
                page_num=global_page_num,
                text=cleaned,
                tables=[],
                is_ocr=use_ocr,
            ))
        else:
            failed_local_indices.append(local_idx)

    return pages, failed_local_indices


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def extract_pdf(pdf_bytes: bytes) -> ExtractionResult:
    """
    Extract text from a PDF using Docling, in BATCH_SIZE-page batches.

    For each batch:
      Pass 1 — Docling StandardPipeline without OCR (full layout, tables, formulas).
      Pass 2 — Docling StandardPipeline with OCR    (only pages that failed Pass 1).
      Pass 3 — PyMuPDF rasterise → RapidOCR         (only pages that failed Pass 2).
      Pass 4 — Docling SimplePipeline               (only pages that failed Pass 3).
      VRAM is flushed after every Docling pass via del converter + torch.cuda.empty_cache().

    All batches are merged into a single ExtractionResult in page order.
    """
    if not pdf_bytes:
        return ExtractionResult(error="Empty PDF bytes provided.")

    total_pages = _get_total_pages(pdf_bytes)
    if total_pages == 0:
        return ExtractionResult(error="Could not determine page count (PyMuPDF failed).")

    num_batches = (total_pages + BATCH_SIZE - 1) // BATCH_SIZE
    logger.info(
        "[EXTRACT] Total pages: %d | Batch size: %d | Batches: %d",
        total_pages, BATCH_SIZE, num_batches,
    )

    all_pages: List[PageResult] = []
    used_ocr_globally = False

    for batch_num in range(num_batches):
        start_0 = batch_num * BATCH_SIZE
        end_0   = min(start_0 + BATCH_SIZE - 1, total_pages - 1)  # 0-based inclusive
        batch_page_count = end_0 - start_0 + 1

        logger.info(
            "[EXTRACT] Batch %d/%d — pages %d–%d (1-based)",
            batch_num + 1, num_batches, start_0 + 1, end_0 + 1,
        )

        # Slice this batch from the full PDF (PyMuPDF, bytes only)
        batch_bytes = _slice_pdf_pages(pdf_bytes, start_0, end_0)
        if batch_bytes is None:
            logger.warning(
                "[EXTRACT] Skipping batch %d — could not slice pages %d–%d.",
                batch_num + 1, start_0 + 1, end_0 + 1,
            )
            continue

        # Track which pages are recovered. Key = global 1-based page number.
        pages_this_batch: Dict[int, PageResult] = {}

        # ── Pass 1: Docling StandardPipeline without OCR ──────────────────────
        pages_p1, failed_local = _run_docling_on_bytes(
            batch_bytes,
            global_page_offset=start_0,
            use_ocr=False,
            simple=False,
        )
        for p in pages_p1:
            pages_this_batch[p.page_num] = p

        logger.info(
            "[EXTRACT] Batch %d Pass 1: %d/%d pages extracted, %d failed.",
            batch_num + 1, len(pages_p1), batch_page_count, len(failed_local),
        )

        # ── Pass 2: Docling StandardPipeline with OCR (failed pages only) ─────
        if failed_local:
            logger.info(
                "[EXTRACT] Batch %d Pass 2 (OCR): %d page(s) to recover.",
                batch_num + 1, len(failed_local),
            )
            # Slice only the failed pages for pass 2
            failed_global_nums = [start_0 + idx + 1 for idx in failed_local]
            recovered_p2 = 0
            for global_num in failed_global_nums:
                local_0 = global_num - start_0 - 1  # 0-based within batch
                page_bytes = _slice_pdf_pages(pdf_bytes, global_num - 1, global_num - 1)
                if page_bytes is None:
                    continue
                pages_single, _ = _run_docling_on_bytes(
                    page_bytes,
                    global_page_offset=global_num - 1,
                    use_ocr=True,
                    simple=False,
                )
                for p in pages_single:
                    if p.page_num not in pages_this_batch:
                        pages_this_batch[p.page_num] = p
                        recovered_p2 += 1
                        used_ocr_globally = True

            still_failed_p2 = [
                idx for idx in failed_local
                if (start_0 + idx + 1) not in pages_this_batch
            ]
            logger.info(
                "[EXTRACT] Batch %d Pass 2: recovered %d page(s), still failed: %d.",
                batch_num + 1, recovered_p2, len(still_failed_p2),
            )

            # ── Pass 3: PyMuPDF rasterise → RapidOCR (failed pages only) ──────
            if still_failed_p2:
                logger.info(
                    "[EXTRACT] Batch %d Pass 3 (rasterise+OCR): %d page(s) to recover.",
                    batch_num + 1, len(still_failed_p2),
                )
                pages_p3 = _rasterise_and_ocr(
                    batch_bytes,
                    failed_local_indices=still_failed_p2,
                    global_page_offset=start_0,
                )
                recovered_p3 = 0
                for p in pages_p3:
                    if p.page_num not in pages_this_batch:
                        pages_this_batch[p.page_num] = p
                        recovered_p3 += 1
                        used_ocr_globally = True

                still_failed_p3 = [
                    idx for idx in still_failed_p2
                    if (start_0 + idx + 1) not in pages_this_batch
                ]
                logger.info(
                    "[EXTRACT] Batch %d Pass 3: recovered %d page(s), still failed: %d.",
                    batch_num + 1, recovered_p3, len(still_failed_p3),
                )

                # ── Pass 4: Docling SimplePipeline (failed pages only) ─────────
                if still_failed_p3:
                    logger.info(
                        "[EXTRACT] Batch %d Pass 4 (SimplePipeline): %d page(s) to recover.",
                        batch_num + 1, len(still_failed_p3),
                    )
                    recovered_p4 = 0
                    for local_idx in still_failed_p3:
                        global_num = start_0 + local_idx + 1
                        page_bytes = _slice_pdf_pages(pdf_bytes, global_num - 1, global_num - 1)
                        if page_bytes is None:
                            continue
                        pages_single, _ = _run_docling_on_bytes(
                            page_bytes,
                            global_page_offset=global_num - 1,
                            use_ocr=False,
                            simple=True,
                        )
                        for p in pages_single:
                            if p.page_num not in pages_this_batch:
                                pages_this_batch[p.page_num] = p
                                recovered_p4 += 1

                    still_failed_p4 = [
                        idx for idx in still_failed_p3
                        if (start_0 + idx + 1) not in pages_this_batch
                    ]
                    logger.info(
                        "[EXTRACT] Batch %d Pass 4: recovered %d page(s), still failed: %d.",
                        batch_num + 1, recovered_p4, len(still_failed_p4),
                    )

        # Append batch results in page order
        for pnum in sorted(pages_this_batch):
            all_pages.append(pages_this_batch[pnum])

    if not all_pages:
        return ExtractionResult(
            error="Docling produced no pages after all batches. PDF may be corrupt or unsupported."
        )

    extracted_page_nums = {p.page_num for p in all_pages}
    failed_pages = sorted(set(range(1, total_pages + 1)) - extracted_page_nums)
    failed_note = (
        f"Pages {failed_pages} could not be extracted after all recovery attempts."
        if failed_pages else None
    )

    logger.info(
        "[EXTRACT] Done: %d/%d pages extracted. OCR used: %s",
        len(all_pages), total_pages, used_ocr_globally,
    )

    return ExtractionResult(
        pages=all_pages,
        total_pages=total_pages,
        used_ocr=used_ocr_globally,
        error=None,
        failed_pages_note=failed_note,
    )
