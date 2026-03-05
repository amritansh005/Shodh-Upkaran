from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SessionState:
    # Current topic/query for this session (last "search" topic)
    current_query: Optional[str] = None

    # Pagination cursor for what the user is currently viewing (0-based offset)
    cursor_start: int = 0

    # Page size used when slicing results for display
    page_size: int = 10

    # Total results reported by backend for the current query (may be None until first search)
    total_results: Optional[int] = None

    # A progressively loaded window of results for the current query.
    # This grows as the user pages forward (chunking).
    prefetched_results: List[Dict[str, Any]] = field(default_factory=list)

    # The last page slice shown to the user (used for "open 3" within the page)
    last_results: List[Dict[str, Any]] = field(default_factory=list)

    # Any additional metadata you already keep
    meta: Dict[str, Any] = field(default_factory=dict)

    # Active paper for Q&A (set when user does "open <n>")
    active_paper_arxiv_id: str = ""
    active_paper_title: str = ""


class InMemorySessionStore:
    def __init__(self) -> None:
        self._states: Dict[str, SessionState] = {}

    def get(self, session_id: str) -> SessionState:
        if session_id not in self._states:
            self._states[session_id] = SessionState()
        return self._states[session_id]

    def reset(self, session_id: str) -> SessionState:
        self._states[session_id] = SessionState()
        return self._states[session_id]