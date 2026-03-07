"""
section_assembler.py — Data models for the section assembly pipeline.

SectionAssembly and PaperSection are used by:
  • vision_extractor.py  — builds SectionAssembly from GPT-4o Vision results
  • chunker.py           — chunks PaperSection objects into embedddable Chunk objects
  • paper_store.py       — persists PaperSection objects to PostgreSQL

The old PyMuPDF bounding-box assemble_sections() function has been removed.
All section assembly is now done via GPT-4o Vision in vision_extractor.py.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class PaperSection:
    section_index: int
    heading_level: int
    heading_text: str
    parent_heading: Optional[str]
    page_start: int
    page_end: int
    content_text: str
    content_length: int


@dataclass
class SectionAssembly:
    sections: List[PaperSection] = field(default_factory=list)
    error: Optional[str] = None

    def is_empty(self) -> bool:
        return not self.sections
