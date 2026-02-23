from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


class ChatService:
    """
    Progressive prefetch (chunking) design:

    - On SEARCH: fetch prefetch_max_results once (e.g., 200) and cache in session.
    - On NEXT / OPEN beyond loaded: fetch next chunks (prefetch_chunk_size) and append.
    - For any page within already-loaded results: slice locally, no backend call.

    This prevents repeated arXiv calls for pagination within already-fetched data.
    """

    def __init__(
        self,
        arxiv,
        llm,
        store,
        page_size_default: int = 10,
        prefetch_max_results: int = 200,
        prefetch_chunk_size: int = 200,
        hard_total_cap: int = 5000,
    ) -> None:
        self.arxiv = arxiv
        self.llm = llm
        self.store = store

        self.page_size_default = max(1, int(page_size_default))
        self.prefetch_max_results = max(1, int(prefetch_max_results))
        self.prefetch_chunk_size = max(1, int(prefetch_chunk_size))
        self.hard_total_cap = max(100, int(hard_total_cap))

    # -----------------------------
    # Public entrypoint (LLM-only routing)
    # -----------------------------
    async def handle_message(
        self, session_id: str, message: str
    ) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
        state = self.store.get(session_id)
        msg = (message or "").strip()

        if not msg:
            return "Say something 🙂 (try: `search ai in healthcare`)", [], self._meta(state)

        # LLM-only routing + slot extraction (NO REGEX)
        intent = self.llm.parse_intent(
            user_message=msg,
            session_state=self._meta(state),
        )

        action = (intent.get("action") or "chat").strip().lower()
        topic = (intent.get("topic") or "").strip()
        index = intent.get("index", None)
        arxiv_id = (intent.get("arxiv_id") or "").strip()
        chat_response = (intent.get("chat_response") or "").strip()

        if action == "help":
            return self._help_text(), [], self._meta(state)

        if action == "reset":
            self.store.reset(session_id)
            state = self.store.get(session_id)
            return "Reset done. You can `search <topic>` again.", [], self._meta(state)

        if action == "search":
            if not topic:
                return "Please provide a topic. Example: `search ai in healthcare`", [], self._meta(state)
            return await self._do_search(state, topic)

        # The remaining actions require an active search context
        if action in ("next", "open", "paper") and not state.current_query:
            return (
                "No active search yet. Use: `search <topic>` (example: `search transformers in nlp`).",
                [],
                self._meta(state),
            )

        if action == "next":
            return await self._do_next(state)

        if action == "open":
            if index is None:
                return "Usage: `open 5` (opens the 5th paper on the current page).", [], self._meta(state)
            try:
                idx_int = int(index)
            except Exception:
                return "Usage: `open 5` (opens the 5th paper on the current page).", [], self._meta(state)
            return await self._do_open_index(state, idx_int)

        if action == "paper":
            if not arxiv_id:
                return "Usage: `paper <arxiv_id>` (example: `paper 2003.10303`)", [], self._meta(state)
            return await self._do_paper(state, arxiv_id)

        # chat / small-talk
        if action == "chat":
            if chat_response:
                return chat_response, [], self._meta(state)
            return "Tell me a topic to search (example: `ai in healthcare`) or type `help`.", [], self._meta(state)

        # Safety fallback
        return "Unknown request. Type `help` to see allowed actions.", [], self._meta(state)

    # -----------------------------
    # Core behaviors
    # -----------------------------
    async def _do_search(
        self, state, query: str
    ) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
        state.current_query = query.strip()
        state.cursor_start = 0
        state.page_size = self.page_size_default
        state.total_results = None
        state.prefetched_results = []
        state.last_results = []

        # Initial prefetch
        fetched, total = await self._fetch_from_backend(
            topic=state.current_query,
            start=0,
            max_results=min(self.prefetch_max_results, self.hard_total_cap),
        )
        state.prefetched_results = fetched
        state.total_results = total

        page = self._slice_page(state, start=0)
        state.last_results = page

        reply = self._format_list_reply(
            state=state,
            start=0,
            page=page,
            note="(cached — use `next` or `open <n>`)",
        )
        return reply, page, self._meta(state)

    async def _do_next(self, state) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
        next_start = state.cursor_start + state.page_size

        # If total known and we're already at/over end
        if state.total_results is not None and next_start >= state.total_results:
            return "No more results.", [], self._meta(state)

        # Ensure enough results are loaded locally for the next page
        needed = next_start + state.page_size
        print(
            f"[CHATBOT] next requested -> cursor={state.cursor_start} "
            f"page_size={state.page_size} needed={needed} loaded={len(state.prefetched_results)}"
        )
        await self._ensure_loaded(state, needed=needed)

        # If still not enough (e.g., backend returned fewer than requested)
        if next_start >= len(state.prefetched_results):
            return "No more results.", [], self._meta(state)

        state.cursor_start = next_start
        page = self._slice_page(state, start=state.cursor_start)
        state.last_results = page

        reply = self._format_list_reply(
            state=state,
            start=state.cursor_start,
            page=page,
            note="(cached)" if len(page) else "",
        )
        return reply, page, self._meta(state)

    async def _do_open_index(
        self, state, idx: int
    ) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
        """
        LLM provides idx directly (no regex). Behavior:
        - If idx fits within current page (state.last_results): open that.
        - Else treat idx as global index across entire search and load as needed.
        """
        if idx is None or idx <= 0:
            return "Index must be >= 1. Example: `open 2`", [], self._meta(state)

        # First interpret as "open Nth on current page"
        if state.last_results and 1 <= idx <= len(state.last_results):
            print(f"[CHATBOT] open served from CURRENT PAGE cache (idx={idx}, cursor_start={state.cursor_start})")
            paper = state.last_results[idx - 1]
            return self._format_paper_reply(paper), [paper], self._meta(state)

        # Otherwise interpret as "open Nth overall in this search"
        global_index = idx - 1
        print(f"[CHATBOT] open needs GLOBAL cache (idx={idx}) -> ensure_loaded({global_index + 1})")
        await self._ensure_loaded(state, needed=global_index + 1)

        if global_index >= len(state.prefetched_results):
            return "That index is beyond the results I have.", [], self._meta(state)

        paper = state.prefetched_results[global_index]
        return self._format_paper_reply(paper), [paper], self._meta(state)

    async def _do_paper(
        self, state, arg: str
    ) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
        arxiv_id = (arg or "").strip()
        if not arxiv_id:
            return "Usage: `paper <arxiv_id>` (example: `paper 2003.10303`)", [], self._meta(state)

        # Delegate to backend if you have a "paper" endpoint; otherwise reuse open-from-cache if present.
        # We'll attempt backend first (more reliable).
        try:
            paper = await self.arxiv.get_paper(arxiv_id)
            if not paper:
                return f"No paper found for arXiv id: {arxiv_id}", [], self._meta(state)

            paper_dict = self._paper_to_dict(paper)
            return self._format_paper_reply(paper_dict), [paper_dict], self._meta(state)

        except Exception:
            # Fallback: search within cached results
            for p in state.prefetched_results:
                pid = str(p.get("arxiv_id") or p.get("id") or "")
                if pid == arxiv_id:
                    return self._format_paper_reply(p), [p], self._meta(state)
            return f"No paper found for arXiv id: {arxiv_id}", [], self._meta(state)

    # -----------------------------
    # Progressive prefetch (chunking)
    # -----------------------------
    async def _ensure_loaded(self, state, needed: int) -> None:
        """
        Ensure at least `needed` items exist in state.prefetched_results.
        Fetch additional chunks from backend only if required.
        """
        if needed <= len(state.prefetched_results):
            return

        # Don’t load beyond safety cap
        if len(state.prefetched_results) >= self.hard_total_cap:
            return

        # If total is known, don’t load beyond it
        if state.total_results is not None and len(state.prefetched_results) >= state.total_results:
            return

        # Fetch chunks until we satisfy `needed` or cannot fetch more
        while len(state.prefetched_results) < needed:
            start = len(state.prefetched_results)

            # Determine how much to request this round
            remaining_cap = self.hard_total_cap - start
            chunk = min(self.prefetch_chunk_size, remaining_cap)

            # If total known, also cap to remaining total
            if state.total_results is not None:
                remaining_total = max(0, state.total_results - start)
                chunk = min(chunk, remaining_total)

            if chunk <= 0:
                return

            fetched, total = await self._fetch_from_backend(
                topic=state.current_query,
                start=start,
                max_results=chunk,
            )

            # Update total if backend provides it
            if state.total_results is None and total is not None:
                state.total_results = total

            # If backend returns nothing, stop
            if not fetched:
                return

            state.prefetched_results.extend(fetched)

            # If backend returned fewer than requested, likely end reached
            if len(fetched) < chunk:
                return

    def _paper_to_dict(self, p: Any) -> Dict[str, Any]:
        """
        Normalize backend/client objects to dict so `.get()` works everywhere.
        Supports:
          - dict
          - Pydantic v2 models (model_dump)
          - Pydantic v1 models (dict)
        """
        if isinstance(p, dict):
            return p

        if hasattr(p, "model_dump"):
            try:
                return p.model_dump()
            except Exception:
                pass

        if hasattr(p, "dict"):
            try:
                return p.dict()
            except Exception:
                pass

        try:
            return dict(p)
        except Exception:
            return {"title": str(p)}

    async def _fetch_from_backend(
        self, topic: str, start: int, max_results: int
    ) -> Tuple[List[Dict[str, Any]], Optional[int]]:
        """
        We keep it defensive so your existing backend response shape still works.
        Also normalizes results into dicts (Paper -> dict).
        """
        print(f"[CHATBOT] backend.search(topic={topic!r}, start={start}, max_results={max_results})")
        resp = await self.arxiv.search(topic=topic, start=start, max_results=max_results)

        if resp is None:
            return [], None

        # dict response: {"results": [...], "total_results": 1234, ...}
        if isinstance(resp, dict):
            raw_results = resp.get("results") or resp.get("items") or resp.get("papers") or []
            total = resp.get("total_results") or resp.get("total") or resp.get("count")
            try:
                total = int(total) if total is not None else None
            except Exception:
                total = None

            if not isinstance(raw_results, list):
                return [], total

            results = [self._paper_to_dict(p) for p in raw_results]
            return results, total

        # list response: [Paper|dict...]
        if isinstance(resp, list):
            results = [self._paper_to_dict(p) for p in resp]
            return results, None

        return [], None

    # -----------------------------
    # Utilities: formatting
    # -----------------------------
    def _slice_page(self, state, start: int) -> List[Dict[str, Any]]:
        end = start + state.page_size
        return state.prefetched_results[start:end]

    def _format_list_reply(self, state, start: int, page: List[Dict[str, Any]], note: str = "") -> str:
        if not page:
            return "No results found."

        total_str = "?"
        if state.total_results is not None:
            total_str = str(state.total_results)

        shown_from = start + 1
        shown_to = start + len(page)
        loaded = len(state.prefetched_results)

        lines = []
        lines.append(f"Showing {shown_from}-{shown_to} of {total_str} for **{state.current_query}** {note}".strip())
        lines.append(f"(loaded in session cache: {loaded})")
        for i, p in enumerate(page, start=1):
            title = (p.get("title") or "").strip() or "Untitled"
            arxiv_id = (p.get("arxiv_id") or p.get("id") or "").strip()
            if arxiv_id:
                lines.append(f"{i}) {title} — {arxiv_id}")
            else:
                lines.append(f"{i}) {title}")
        return "\n".join(lines)

    def _format_paper_reply(self, p: Dict[str, Any]) -> str:
        title = (p.get("title") or "").strip() or "Untitled"
        arxiv_id = (p.get("arxiv_id") or p.get("id") or "").strip()
        authors = p.get("authors") or p.get("author") or []
        if isinstance(authors, list):
            authors_str = ", ".join([str(a) for a in authors][:8])
        else:
            authors_str = str(authors)

        summary = (p.get("summary") or p.get("abstract") or "").strip()
        if len(summary) > 900:
            summary = summary[:900].rstrip() + "..."

        lines = [f"**{title}**"]
        if arxiv_id:
            lines.append(f"arXiv: {arxiv_id}")
        if authors_str:
            lines.append(f"Authors: {authors_str}")
        if summary:
            lines.append("")
            lines.append(summary)
        return "\n".join(lines)

    def _help_text(self) -> str:
        return (
            "Actions allowed:\n"
            "- `help`\n"
            "- `reset`\n"
            "- `search <topic>`\n"
            "- `next`\n"
            "- `open <n>` (opens the nth paper on current page; if not on page, tries nth overall)\n"
            "- `paper <arxiv_id>`\n"
        )

    def _meta(self, state) -> Dict[str, Any]:
        return {
            "query": state.current_query,
            "cursor_start": state.cursor_start,
            "page_size": state.page_size,
            "loaded": len(state.prefetched_results),
            "total_results": state.total_results,
            "prefetch_max_results": self.prefetch_max_results,
            "prefetch_chunk_size": self.prefetch_chunk_size,
            "hard_total_cap": self.hard_total_cap,
        }