"""
chunker.py — Token-aware sliding window text chunker.

Splits extracted PDF pages (or assembled sections) into overlapping chunks
suitable for embedding. Respects paragraph → sentence → word boundary order
when splitting. Tracks source page number and section heading per chunk.

Two entry points:
  chunk_sections(sections) — preferred: chunks from section_assembler output,
                             each chunk tagged with its section heading.
  chunk_pages(pages)       — fallback: chunks from flat pdf_extractor output,
                             section_heading is None for all chunks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

# Target ~600 tokens per chunk, ~100 token overlap
# Approximated as characters (avg 4 chars/token for English)
CHUNK_SIZE      = 600   # tokens
OVERLAP         = 100   # tokens
CHARS_PER_TOKEN = 4

@dataclass
class Chunk:
    chunk_index:     int
    page_num:        Optional[int]   # 1-based source page, None if mixed
    text:            str
    section_heading: Optional[str]   # heading this chunk belongs to, or None


def chunk_sections(sections) -> List[Chunk]:
    """
    Chunk a list of PaperSection objects into globally-indexed Chunks.
    Each chunk is tagged with its section heading for section-aware retrieval.

    Uses the same sliding window for all sections regardless of length.
    The heading text is prepended to every chunk so the embedding captures
    both section context and content together.
    """
    all_chunks: List[Chunk] = []
    global_idx = 0

    for section in sections:
        text = (section.content_text or "").strip()
        if not text:
            continue

        heading  = section.heading_text
        page_num = section.page_start

        raw_chunks = _chunk_text(text, page_num=page_num)
        for rc in raw_chunks:
            chunk_text = f"[{heading}]\n\n{rc.text}"
            all_chunks.append(Chunk(
                chunk_index     = global_idx,
                page_num        = rc.page_num,
                text            = chunk_text,
                section_heading = heading,
            ))
            global_idx += 1

    return all_chunks


def chunk_pages(pages) -> List[Chunk]:
    """
    Chunk a list of PageResult objects into globally-indexed Chunks.
    Fallback used when section assembly is unavailable.
    section_heading is None for all chunks produced here.
    """
    all_chunks: List[Chunk] = []
    global_idx = 0

    for page in pages:
        parts = [page.text or ""]
        for table in (page.tables or []):
            if table.strip():
                parts.append(table)
        combined = "\n\n".join(p for p in parts if p.strip())

        for chunk in _chunk_text(combined, page_num=page.page_num):
            all_chunks.append(Chunk(
                chunk_index     = global_idx,
                page_num        = chunk.page_num,
                text            = chunk.text,
                section_heading = None,
            ))
            global_idx += 1

    return all_chunks


def _chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = OVERLAP,
    page_num: Optional[int] = None,
) -> List[Chunk]:
    text = (text or "").strip()
    if not text:
        return []

    max_chars     = chunk_size * CHARS_PER_TOKEN
    overlap_chars = overlap * CHARS_PER_TOKEN
    chunks: List[Chunk] = []
    start = 0
    idx   = 0

    while start < len(text):
        end = start + max_chars

        if end >= len(text):
            chunk_str = text[start:].strip()
            if chunk_str:
                chunks.append(Chunk(
                    chunk_index=idx, page_num=page_num,
                    text=chunk_str, section_heading=None,
                ))
            break

        # Try to split on paragraph, then sentence, then word boundary
        split_pos = (
            _find_split(text, end, window=400, sep="\n\n")
            or _find_split(text, end, window=200, sep=". ")
            or _find_split(text, end, window=100, sep=" ")
            or end
        )

        chunk_str = text[start:split_pos].strip()
        if chunk_str:
            chunks.append(Chunk(
                chunk_index=idx, page_num=page_num,
                text=chunk_str, section_heading=None,
            ))
            idx += 1

        start = max(start + 1, split_pos - overlap_chars)

    return chunks


def _find_split(text: str, near: int, window: int, sep: str) -> Optional[int]:
    """Search backwards from `near` within `window` chars for `sep`."""
    search_start = max(0, near - window)
    segment = text[search_start:near]
    pos = segment.rfind(sep)
    if pos == -1:
        return None
    return search_start + pos + len(sep)
