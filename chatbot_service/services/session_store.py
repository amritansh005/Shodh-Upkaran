from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from chatbot_service.schemas import Paper


@dataclass
class SessionState:
    last_query: Optional[str] = None
    last_start: int = 0
    page_size: int = 10
    last_results: List[Paper] = field(default_factory=list)


class InMemorySessionStore:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._store: Dict[str, SessionState] = {}

    async def get(self, session_id: str) -> SessionState:
        async with self._lock:
            if session_id not in self._store:
                self._store[session_id] = SessionState()
            return self._store[session_id]

    async def clear(self, session_id: str) -> None:
        async with self._lock:
            self._store.pop(session_id, None)
