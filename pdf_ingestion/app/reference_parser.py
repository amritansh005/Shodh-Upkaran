from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class ParsedReference:
    index: Optional[int]
    raw_text: str


@dataclass
class ReferenceParseResult:
    style: str
    count: int
    entries: List[ParsedReference]
    confidence: float
    last_reference_number: Optional[int]

    def to_metadata(self) -> Optional[Dict[str, Any]]:
        if self.count <= 0:
            return None
        return {
            "reference_numbering_style": self.style,
            "reference_count": self.count,
            "last_reference_number": self.last_reference_number,
            "reference_parse_confidence": self.confidence,
        }


_HEADING_PATTERNS = [
    re.compile(r"^(?:[ivxlcdm]+)[.)]\s+", re.IGNORECASE),
    re.compile(r"^\d+[.)]\s+"),
    re.compile(r"^\d+\s+"),
    re.compile(r"^[A-Za-z][.)]\s+"),
]

_BRACKET_START_RE = re.compile(r"^\s*\[(\d{1,4})\]\s+\S")
_BRACKET_ANYWHERE_RE = re.compile(r"(?<!\w)\[(\d{1,4})\]\s+")
_DOT_START_RE = re.compile(r"^\s*(\d{1,4})\.\s+\S")
_PAREN_START_RE = re.compile(r"^\s*(\d{1,4})\)\s+\S")
_AUTHOR_YEAR_START_RE = re.compile(
    r"^\s*(?:[A-Z][A-Za-z'`.-]+(?:,|\s+and\s+|\s*&\s+|\s+et al\.?\s*,?\s*)).{0,80}?(?:\(|,\s*)(19|20)\d{2}(?:\)|[a-z]?\b)"
)
_YEAR_ONLY_RE = re.compile(r"^\s*(19|20)\d{2}[a-z]?\.?\s*$")
_PAGE_ONLY_RE = re.compile(r"^\s*(?:pp?\.?\s*)?\d+(?:\s*[-–—]\s*\d+)?\.?\s*$", re.IGNORECASE)
_VOLUME_ONLY_RE = re.compile(r"^\s*(?:vol\.?|no\.?|issue)\s+\d+\.?\s*$", re.IGNORECASE)
_DOI_ONLY_RE = re.compile(r"^\s*(?:doi\s*:|https?://(?:dx\.)?doi\.org/)\S+\s*$", re.IGNORECASE)

REFERENCE_TERMS = (
    "reference",
    "bibliography",
    "works cited",
    "reference list",
    "references and notes",
    "literature cited",
)


def find_reference_section(sections: Sequence[Any]) -> Optional[Any]:
    for section in sections:
        heading = (getattr(section, "heading_text", "") or "").strip().lower()
        for pattern in _HEADING_PATTERNS:
            heading = pattern.sub("", heading).strip()
        if any(term in heading for term in REFERENCE_TERMS):
            return section
    return None


def parse_reference_metadata(reference_text: str) -> Optional[Dict[str, Any]]:
    result = parse_references(reference_text)
    return result.to_metadata() if result else None


def parse_references(reference_text: str) -> Optional[ReferenceParseResult]:
    if not reference_text or not reference_text.strip():
        return None

    text = reference_text.replace("\r\n", "\n").replace("\r", "\n")
    lines = _normalize_lines(text)
    if not lines:
        return None

    style = _detect_style(lines, text)

    if style in {"bracket", "plain-dot", "plain-paren"}:
        result = _parse_numbered(text, lines, style)
        if result and result.count > 0:
            return result

    if style in {"author-year", "unknown"}:
        result = _parse_author_year(lines)
        if result and result.count > 0:
            return result

    return None


def _normalize_lines(text: str) -> List[str]:
    lines: List[str] = []
    for raw in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        cleaned = re.sub(r"\s+", " ", raw).strip()
        if cleaned:
            lines.append(cleaned)
    return lines


def _detect_style(lines: Sequence[str], text: str) -> str:
    sample = list(lines[:80])
    bracket = max(
        sum(1 for line in sample if _BRACKET_START_RE.match(line)),
        len(_BRACKET_ANYWHERE_RE.findall(text[:20000])),
    )
    dot = sum(1 for line in sample if _DOT_START_RE.match(line))
    paren = sum(1 for line in sample if _PAREN_START_RE.match(line))
    author_year = sum(1 for line in sample if _looks_like_author_year_start(line))

    scores = {
        "bracket": bracket,
        "plain-dot": dot,
        "plain-paren": paren,
        "author-year": author_year,
    }
    best_style = max(scores, key=scores.get)
    best_score = scores[best_style]
    return best_style if best_score >= 2 else "unknown"


def _parse_numbered(text: str, lines: Sequence[str], style: str) -> Optional[ReferenceParseResult]:
    if style == "bracket":
        return _parse_bracket_numbered(text)

    start_re = {
        "plain-dot": _DOT_START_RE,
        "plain-paren": _PAREN_START_RE,
    }[style]

    entries: List[ParsedReference] = []
    current_index: Optional[int] = None
    current_parts: List[str] = []
    numbers_in_order: List[int] = []

    for line in lines:
        m = start_re.match(line)
        if m:
            idx = int(m.group(1))
            if _is_implausible_reference_number(idx):
                if current_parts:
                    current_parts.append(line)
                continue
            if current_parts:
                entries.append(ParsedReference(index=current_index, raw_text=" ".join(current_parts).strip()))
            current_index = idx
            current_parts = [line]
            numbers_in_order.append(idx)
        elif current_parts:
            current_parts.append(line)

    if current_parts:
        entries.append(ParsedReference(index=current_index, raw_text=" ".join(current_parts).strip()))

    if not entries:
        return None

    valid_numbers = _longest_consecutive_run(numbers_in_order)
    if not valid_numbers:
        return None

    valid_set = set(valid_numbers)
    filtered_entries = [e for e in entries if e.index in valid_set]
    if not filtered_entries:
        return None

    return ReferenceParseResult(
        style=style,
        count=len(valid_numbers),
        entries=filtered_entries,
        confidence=_numbered_confidence(numbers_in_order, valid_numbers),
        last_reference_number=valid_numbers[-1],
    )


def _parse_bracket_numbered(text: str) -> Optional[ReferenceParseResult]:
    normalized = re.sub(r"[ \t]+", " ", text.replace("\r\n", "\n").replace("\r", "\n"))
    matches = list(_BRACKET_ANYWHERE_RE.finditer(normalized))
    if not matches:
        return None

    entries: List[ParsedReference] = []
    numbers_in_order: List[int] = []

    for i, match in enumerate(matches):
        idx = int(match.group(1))
        if _is_implausible_reference_number(idx):
            continue
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(normalized)
        raw = re.sub(r"\s+", " ", normalized[start:end]).strip()
        if not raw:
            continue
        numbers_in_order.append(idx)
        entries.append(ParsedReference(index=idx, raw_text=raw))

    if not entries:
        return None

    valid_numbers = _longest_consecutive_run(numbers_in_order)
    if not valid_numbers:
        return None

    valid_set = set(valid_numbers)
    filtered_entries = [e for e in entries if e.index in valid_set]
    if not filtered_entries:
        return None

    return ReferenceParseResult(
        style="bracket",
        count=len(valid_numbers),
        entries=filtered_entries,
        confidence=_numbered_confidence(numbers_in_order, valid_numbers),
        last_reference_number=valid_numbers[-1],
    )


def _parse_author_year(lines: Sequence[str]) -> Optional[ReferenceParseResult]:
    entries: List[ParsedReference] = []
    current_parts: List[str] = []

    for line in lines:
        if _looks_like_author_year_start(line):
            if current_parts:
                entries.append(ParsedReference(index=None, raw_text=" ".join(current_parts).strip()))
            current_parts = [line]
        elif current_parts:
            current_parts.append(line)

    if current_parts:
        entries.append(ParsedReference(index=None, raw_text=" ".join(current_parts).strip()))

    entries = [e for e in entries if len(e.raw_text) >= 20]
    if not entries:
        return None

    confidence = min(0.95, 0.55 + 0.01 * len(entries))
    return ReferenceParseResult(
        style="author-year",
        count=len(entries),
        entries=entries,
        confidence=confidence,
        last_reference_number=None,
    )


def _looks_like_author_year_start(line: str) -> bool:
    if _YEAR_ONLY_RE.match(line) or _PAGE_ONLY_RE.match(line) or _VOLUME_ONLY_RE.match(line) or _DOI_ONLY_RE.match(line):
        return False
    return bool(_AUTHOR_YEAR_START_RE.match(line))


def _is_implausible_reference_number(idx: int) -> bool:
    return idx <= 0 or idx >= 5000


def _longest_consecutive_run(numbers: Sequence[int]) -> List[int]:
    if not numbers:
        return []

    cleaned: List[int] = []
    seen = set()
    for n in numbers:
        if n in seen:
            continue
        seen.add(n)
        cleaned.append(n)

    best_run: List[int] = []
    current_run: List[int] = []

    for n in cleaned:
        if not current_run:
            current_run = [n]
            continue
        last = current_run[-1]
        if n == last + 1:
            current_run.append(n)
        else:
            if len(current_run) > len(best_run):
                best_run = current_run[:]
            current_run = [n]

    if len(current_run) > len(best_run):
        best_run = current_run[:]

    if len(best_run) >= 2:
        return best_run
    return [cleaned[0]] if cleaned else []


def _numbered_confidence(all_numbers: Sequence[int], valid_numbers: Sequence[int]) -> float:
    if not all_numbers or not valid_numbers:
        return 0.0
    ratio = len(valid_numbers) / max(1, len(all_numbers))
    sequential_bonus = 0.2 if len(valid_numbers) >= 5 else 0.05
    return min(0.99, 0.65 + 0.25 * ratio + sequential_bonus)