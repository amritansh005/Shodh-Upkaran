"""
vision_extractor.py — PDF section extraction using Marker.

Marker (marker-pdf) uses trained layout detection (surya) + OCR models to
convert any PDF to structured Markdown with ATX headings (# / ## / ###).
Runs on CUDA GPU (RTX 2060) for fast inference.

Install:
    pip install marker-pdf
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

Output types (SectionAssembly, HeadingTree) are identical to the old
vision_extractor — zero downstream changes needed in ingest_pipeline.py,
chunker.py, paper_store.py, or qa_service.py.
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

from pdf_ingestion.app.structural_extractor import HeadingTree, Heading
from pdf_ingestion.app.section_assembler import SectionAssembly, PaperSection


# ---------------------------------------------------------------------------
# Marker model — loaded once at module level, reused across all requests
# ---------------------------------------------------------------------------
# Force CUDA so the RTX 2060 is used. Marker respects the TORCH_DEVICE env
# var, but we also set it explicitly here so it's not left to chance.

os.environ.setdefault("TORCH_DEVICE", "cuda")

_converter = None   # PdfConverter singleton


def _get_converter():
    """
    Lazy-load the Marker PdfConverter and keep it in memory.
    Models are downloaded on first run (~1-2 GB) then cached locally.
    Subsequent calls reuse the already-loaded models.
    """
    global _converter
    if _converter is not None:
        return _converter

    try:
        import torch
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
        from marker.config.parser import ConfigParser

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            logger.warning("[MARKER] CUDA not available — running on CPU (will be slow)")
        else:
            logger.info("[MARKER] Using GPU: %s", torch.cuda.get_device_name(0))

        config_parser = ConfigParser({"output_format": "markdown"})
        _converter = PdfConverter(
            config=config_parser.generate_config_dict(),
            artifact_dict=create_model_dict(),
            processor_list=config_parser.get_processors(),
            renderer=config_parser.get_renderer(),
        )
        logger.info("[MARKER] Models loaded on %s", device.upper())
        return _converter

    except Exception as e:
        logger.error("[MARKER] Failed to load models: %s", e)
        raise


# ---------------------------------------------------------------------------
# Markdown parser — ATX headings → SectionAssembly + HeadingTree
# ---------------------------------------------------------------------------

def _parse_markdown_to_sections(
    markdown: str,
    total_pages: int,
) -> Tuple[SectionAssembly, HeadingTree]:
    """
    Parse Marker's Markdown output into SectionAssembly + HeadingTree.

    ATX headings (# / ## / ###) become PaperSection objects.
    Body text between headings becomes content_text for each section.
    Page numbers are approximated from line position in the Markdown.
    Marker's page separator '---' is used to track page boundaries.
    """
    sections:    List[PaperSection] = []
    headings:    List[Heading]      = []
    level_stack: Dict[int, str]     = {}

    md_lines       = markdown.splitlines()
    total_lines    = max(len(md_lines), 1)
    lines_per_page = max(1, total_lines // max(total_pages, 1))

    current_heading_text:  Optional[str] = None
    current_heading_level: int           = 1
    current_page_start:    int           = 1
    current_body:          List[str]     = []
    section_index:         int           = 0
    current_page:          int           = 1

    def _flush() -> None:
        nonlocal section_index
        if current_heading_text is None:
            return
        content = "\n".join(current_body).strip()
        parent  = None
        for lvl in range(current_heading_level - 1, 0, -1):
            if lvl in level_stack:
                parent = level_stack[lvl]
                break
        sections.append(PaperSection(
            section_index  = section_index,
            heading_level  = current_heading_level,
            heading_text   = current_heading_text,
            parent_heading = parent,
            page_start     = current_page_start,
            page_end       = current_page,
            content_text   = content,
            content_length = len(content),
        ))
        section_index += 1

    preamble_lines:      List[str] = []
    first_heading_found: bool      = False
    preamble_page_start: int       = 1

    for line_idx, raw_line in enumerate(md_lines):
        current_page = max(1, line_idx // lines_per_page + 1)
        line = raw_line.rstrip()

        # Marker uses "---" as a page separator in some output modes
        if line.strip() == "---":
            current_page = min(current_page + 1, total_pages)
            continue

        heading_match = re.match(r"^(#{1,3})\s+(.+)", line)
        if heading_match:
            if not first_heading_found:
                # Flush preamble (title / author / abstract area before first heading)
                preamble_text = "\n".join(preamble_lines).strip()
                if preamble_text:
                    sections.append(PaperSection(
                        section_index  = section_index,
                        heading_level  = 0,
                        heading_text   = "",
                        parent_heading = None,
                        page_start     = preamble_page_start,
                        page_end       = current_page,
                        content_text   = preamble_text,
                        content_length = len(preamble_text),
                    ))
                    section_index += 1
                first_heading_found = True
            else:
                _flush()

            level = len(heading_match.group(1))
            text  = heading_match.group(2).strip()
            # Strip bold/italic markers that sometimes appear inside headings
            text  = re.sub(r"\*{1,2}(.+?)\*{1,2}", r"\1", text).strip()

            current_heading_text  = text
            current_heading_level = level
            current_page_start    = current_page
            current_body          = []

            level_stack[level] = text
            for deeper in list(level_stack.keys()):
                if deeper > level:
                    del level_stack[deeper]

            headings.append(Heading(level=level, text=text, page=current_page))

        else:
            stripped = line.strip()
            if not stripped:
                continue
            if not first_heading_found:
                preamble_lines.append(stripped)
            else:
                current_body.append(stripped)

    _flush()

    # Drop sections with no content (table artefacts etc.) and re-index
    sections = [s for s in sections if s.content_text.strip() or s.heading_level == 0]
    for i, s in enumerate(sections):
        s.section_index = i

    return SectionAssembly(sections=sections), HeadingTree(headings=headings)


# ===========================================================================
# Public entry point — same signature as before, zero downstream changes
# ===========================================================================

def extract_sections_via_vision(
    pdf_bytes: bytes,
    settings=None,      # kept for API compatibility
    max_pages: int = 60,
) -> Tuple[SectionAssembly, HeadingTree, int]:
    """
    Extract sections from a PDF using Marker on GPU (RTX 2060).

    Returns (SectionAssembly, HeadingTree, total_pages).
    Sections → paper_sections table.
    HeadingTree → papers.paper_headings (JSON).
    Both feed chunker → embedder → paper_chunks unchanged.
    """
    if not pdf_bytes:
        err = "Empty PDF bytes."
        return SectionAssembly(error=err), HeadingTree(error=err), 0

    # Get total page count
    total_pages = 0
    try:
        import fitz
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        total_pages = len(doc)
        doc.close()
    except Exception as e:
        err = f"Could not open PDF: {e}"
        return SectionAssembly(error=err), HeadingTree(error=err), 0

    logger.info("[EXTRACTOR] pages=%d  Running Marker on GPU", total_pages)

    # Write PDF to a temp file — Marker expects a file path
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name

        from marker.output import text_from_rendered

        converter = _get_converter()

        # page_range must be a list of ints e.g. [0, 1, 2, 3]
        # Marker 1.10.x does max(self.page_range) < len(doc) — needs actual ints
        pages_to_run = min(max_pages, total_pages)
        converter.config["page_range"] = list(range(pages_to_run))

        rendered = converter(tmp_path)
        markdown, _, _ = text_from_rendered(rendered)

        if not markdown or not markdown.strip():
            err = "Marker returned empty output."
            logger.error("[EXTRACTOR] %s", err)
            return SectionAssembly(error=err), HeadingTree(error=err), total_pages

        assembly, heading_tree = _parse_markdown_to_sections(markdown, total_pages)

        if assembly.is_empty():
            err = "Marker produced no sections."
            logger.error("[EXTRACTOR] %s", err)
            return SectionAssembly(error=err), HeadingTree(error=err), total_pages

        logger.info("[EXTRACTOR] Marker succeeded: %d sections", len(assembly.sections))
        return assembly, heading_tree, total_pages

    except Exception as e:
        err = f"Marker extraction failed: {e}"
        logger.error("[EXTRACTOR] %s", err, exc_info=True)
        return SectionAssembly(error=err), HeadingTree(error=err), total_pages

    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass