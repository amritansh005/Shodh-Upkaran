"""
chunker.py — Token-aware sliding window text chunker.

Splits extracted PDF pages into overlapping chunks suitable for embedding.
Respects paragraph → sentence → word boundary order when splitting.
Tracks source page number per chunk for traceability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

# Target ~600 tokens per chunk, ~100 token overlap
# Approximated as characters (avg 4 chars/token for English)
CHUNK_SIZE    = 600   # tokens
OVERLAP       = 100   # tokens
CHARS_PER_TOKEN = 4


@dataclass
class Chunk:
    chunk_index: int
    page_num: Optional[int]   # 1-based source page, None if mixed
    text: str


def chunk_pages(pages) -> List[Chunk]:
    """
    Chunk a list of PageResult objects into globally-indexed Chunks.
    Each page is chunked independently so page_num stays accurate.
    Tables are appended to the page text before chunking.
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
                chunk_index=global_idx,
                page_num=chunk.page_num,
                text=chunk.text,
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
                chunks.append(Chunk(chunk_index=idx, page_num=page_num, text=chunk_str))
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
            chunks.append(Chunk(chunk_index=idx, page_num=page_num, text=chunk_str))
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
