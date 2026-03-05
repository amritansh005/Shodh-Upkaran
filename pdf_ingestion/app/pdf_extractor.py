"""
pdf_extractor.py — Docling-only PDF extraction with GPU acceleration.

Replaces the previous dual-path (PyMuPDF/pdfplumber + OCR) extractor.

New behavior:
- Always uses IBM Docling to convert the PDF into markdown with page breaks.
- Splits the markdown back into per-page text.
- No fallback. If Docling fails or returns empty content, extraction fails.
- Uses CUDA GPU (RTX 2060) with num_threads=1 for laptop-friendly acceleration.
- Falls back to CPU automatically if CUDA is unavailable.

Notes:
- Tables are included inline in the exported markdown, so `tables=[]` per page.
- `used_ocr` is set to False because we are not explicitly tracking OCR usage here.

Docling 2.75.0 API:
- GPU is configured via PdfFormatOption -> PipelineOptions -> AcceleratorOptions
- Passed through DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(...)})
"""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)

# Token to force page boundary markers in Docling markdown export
_PAGE_BREAK_TOKEN = "\n\n<<<PAGE_BREAK>>>\n\n"


# ------------------------------------------------------------------
# Data model (kept compatible with your existing pipeline)
# ------------------------------------------------------------------

@dataclass
class PageResult:
    page_num: int  # 1-based
    text: str  # extracted text (markdown)
    tables: List[str] = field(default_factory=list)  # kept for compatibility; unused here
    is_ocr: bool = False  # kept for compatibility; Docling-only extractor does not flag OCR


@dataclass
class ExtractionResult:
    pages: List[PageResult] = field(default_factory=list)
    total_pages: int = 0
    used_ocr: bool = False
    error: Optional[str] = None

    def full_text(self) -> str:
        """
        Concatenate page text (tables are expected inline in markdown output).
        Kept compatible with previous behavior.
        """
        parts: List[str] = []
        for p in self.pages:
            if p.text.strip():
                parts.append(p.text)
            for t in p.tables:
                if t.strip():
                    parts.append(t)
        return "\n\n".join(parts)


# ------------------------------------------------------------------
# GPU/CPU accelerator helper (Docling 2.75.0 API)
# ------------------------------------------------------------------

def _build_converter():
    """
    Build a DocumentConverter with GPU (CUDA) if available, else CPU.

    Docling 2.75.0 API:
        DocumentConverter(format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=PipelineOptions(...))
        })

    num_threads=1 keeps GPU usage gentle — suitable for a laptop RTX 2060.
    """
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, AcceleratorOptions
    from docling.datamodel.accelerator_options import AcceleratorDevice

    # Detect CUDA availability at runtime
    try:
        import torch
        cuda_available = torch.cuda.is_available()
    except ImportError:
        cuda_available = False

    if cuda_available:
        device = AcceleratorDevice.CUDA
        logger.info("[EXTRACT] CUDA available — using GPU for Docling.")
    else:
        device = AcceleratorDevice.CPU
        logger.warning("[EXTRACT] CUDA not available — falling back to CPU for Docling.")

    accelerator_options = AcceleratorOptions(
        num_threads=1,   # gentle on laptop GPU; prevents thermal throttle
        device=device,
    )

    pipeline_options = PdfPipelineOptions(
        accelerator_options=accelerator_options,
    )

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )


# ------------------------------------------------------------------
# Main entry point (Docling-only)
# ------------------------------------------------------------------

def extract_pdf(pdf_bytes: bytes) -> ExtractionResult:
    """
    Extract text from PDF bytes using Docling only.

    - Writes PDF bytes to a temporary file.
    - Docling converts the PDF to an internal document model (GPU-accelerated).
    - Export to markdown with explicit page break placeholders.
    - Split into per-page markdown text.
    """
    if not pdf_bytes:
        return ExtractionResult(error="Empty PDF bytes provided.")

    try:
        # IMPORTANT: In Docling v2.x, ImageRefMode is provided by docling-core.
        from docling_core.types.doc import ImageRefMode
    except Exception as e:
        return ExtractionResult(error=f"Docling is not installed or failed to import: {e}")

    try:
        # Use delete=False so Docling can open the file by path on Windows.
        # On Windows, NamedTemporaryFile with delete=True locks the file and
        # prevents other processes (Docling) from opening it -> PermissionError.
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp_path = tmp.name
                tmp.write(pdf_bytes)
                tmp.flush()
            # File is now closed — safe to open by path on Windows

            converter = _build_converter()
            conv = converter.convert(tmp_path, raises_on_error=True)

            doc = getattr(conv, "document", None)
            if doc is None:
                return ExtractionResult(error="Docling conversion returned no document.")

            markdown = doc.export_to_markdown(
                page_break_placeholder=_PAGE_BREAK_TOKEN,
                image_mode=ImageRefMode.PLACEHOLDER,
            )
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

        if not markdown or not markdown.strip():
            return ExtractionResult(error="Docling returned empty content.")

        # Split into per-page blocks
        raw_pages = markdown.split(_PAGE_BREAK_TOKEN)

        # Keep page numbering stable and continuous.
        pages: List[PageResult] = []
        for idx, page_text in enumerate(raw_pages, start=1):
            pages.append(
                PageResult(
                    page_num=idx,
                    text=(page_text or "").strip(),
                    tables=[],    # tables are inline in markdown
                    is_ocr=False, # not tracked here
                )
            )

        if not pages:
            return ExtractionResult(error="Docling produced no pages.")

        # total_pages should reflect the original PDF page count as closely as possible.
        # Docling's markdown split count is used here.
        return ExtractionResult(
            pages=pages,
            total_pages=len(pages),
            used_ocr=False,
            error=None,
        )

    except Exception as e:
        logger.exception("[EXTRACT] Docling extraction failed")
        return ExtractionResult(error=f"Docling extraction failed: {e}")
