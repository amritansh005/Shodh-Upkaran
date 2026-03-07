"""
structural_extractor.py — Data models for the heading/section extraction pipeline.

HeadingTree and Heading are used by:
  • vision_extractor.py  — builds HeadingTree from GPT-4o Vision results
  • qa_service.py        — reads stored HeadingTree for the outline fast-path

The old font-heuristic extract_headings() function has been removed.
All extraction is now done via GPT-4o Vision in vision_extractor.py.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Heading:
    level: int
    text: str
    page: int


@dataclass
class HeadingTree:
    headings: List[Heading] = field(default_factory=list)
    error: Optional[str] = None

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
                Heading(
                    level=int(it["level"]),
                    text=str(it["text"]).strip(),
                    page=int(it["page"]),
                )
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

    def format_for_display(self) -> str:
        lines = []
        for h in self.headings:
            if h.level > 2:
                continue
            indent = "  " * (h.level - 1)
            lines.append(f"{indent}- {h.text}  (page {h.page})")
        return "\n".join(lines)
