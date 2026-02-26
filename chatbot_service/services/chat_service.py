from __future__ import annotations

import difflib
import re
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
        author = (intent.get("author") or "").strip()
        from_year = intent.get("from_year", None)
        to_year = intent.get("to_year", None)
        categories = intent.get("categories") or []
        index = intent.get("index", None)
        title = (intent.get("title") or "").strip()
        arxiv_id = (intent.get("arxiv_id") or "").strip()
        chat_response = (intent.get("chat_response") or "").strip()

        if action == "help":
            return self._help_text(), [], self._meta(state)

        if action == "reset":
            self.store.reset(session_id)
            state = self.store.get(session_id)
            return "Reset done. You can `search <topic>` again.", [], self._meta(state)

        if action == "search":
            if not topic and not author and not categories and from_year is None and to_year is None:
                return (
                    "Please provide at least one search constraint. Examples:\n"
                    "- `ai in healthcare`\n"
                    "- `papers by Andrew Ng`\n"
                    "- `ai in healthcare between 2020 and 2022`\n"
                    "- `cs.AI papers between 2020 and 2022`",
                    [],
                    self._meta(state),
                )
            return await self._do_search(
                state,
                topic=topic,
                author=author,
                from_year=from_year,
                to_year=to_year,
                categories=categories,
            )

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
            # Prefer explicit index when provided
            if index is not None:
                try:
                    idx_int = int(index)
                except Exception:
                    idx_int = None
                if idx_int is None:
                    return "Usage: `open 5` or `open <title>`", [], self._meta(state)
                return await self._do_open_index(state, idx_int)

            # Otherwise try open-by-title
            if title:
                return await self._do_open_title(state, title)

            return "Usage: `open 5` or `open <title>`", [], self._meta(state)

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
        self,
        state,
        topic: str,
        author: str = "",
        from_year: Optional[int] = None,
        to_year: Optional[int] = None,
        categories: Optional[List[str]] = None,
    ) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
        # Human-readable label for the active search (used by NEXT/OPEN context)
        state.current_query = self._describe_search(
            topic=topic,
            author=author,
            from_year=from_year,
            to_year=to_year,
            categories=categories,
        )
        # Keep structured search inputs as well
        state.meta["search"] = {
            "topic": (topic or "").strip(),
            "author": (author or "").strip(),
            "from_year": from_year,
            "to_year": to_year,
            "categories": categories or [],
        }
        state.cursor_start = 0
        state.page_size = self.page_size_default
        state.total_results = None
        state.prefetched_results = []
        state.last_results = []

        # Initial prefetch
        fetched, total = await self._fetch_from_backend(
            topic=(topic or "").strip(),
            author=(author or "").strip(),
            from_year=from_year,
            to_year=to_year,
            categories=categories or [],
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

        if state.total_results is not None and next_start >= state.total_results:
            return "No more results.", [], self._meta(state)

        needed = next_start + state.page_size
        print(
            f"[CHATBOT] next requested -> cursor={state.cursor_start} "
            f"page_size={state.page_size} needed={needed} loaded={len(state.prefetched_results)}"
        )
        await self._ensure_loaded(state, needed=needed)

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
        Behavior:
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

    async def _do_open_title(
        self, state, raw_title: str
    ) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
        """Open a paper by (approximate) title.

        Matching strategy (in order):
        1) Exact normalized title match within currently loaded cache.
        2) Substring match (query contained in title).
        3) Fuzzy match (difflib ratio) within loaded cache.

        If not found and more results may exist, progressively load more chunks
        (up to hard_total_cap / known total) and retry.
        """
        query = (raw_title or "").strip()
        if not query:
            return "Usage: `open <title>`", [], self._meta(state)

        max_rounds = 5
        for _ in range(max_rounds):
            paper, candidates = self._find_best_title_match(state, query)
            if paper is not None:
                return self._format_paper_reply(paper), [paper], self._meta(state)

            if candidates:
                lines = [
                    "I found multiple close title matches. Reply with `open <number>` or paste the exact title:",
                ]
                for i, p in enumerate(candidates[:5], start=1):
                    t = str(p.get("title") or "").strip()
                    pid = str(p.get("arxiv_id") or p.get("id") or "").strip()
                    lines.append(f"{i}) {t} — {pid}")
                return "\n".join(lines), candidates[:5], self._meta(state)

            if state.total_results is not None and len(state.prefetched_results) >= state.total_results:
                break
            if len(state.prefetched_results) >= self.hard_total_cap:
                break

            await self._ensure_loaded(state, needed=len(state.prefetched_results) + self.prefetch_chunk_size)

        return (
            "Couldn't find a paper with that title in the current search results. "
            "Tip: try `next` to browse, or use `paper <arxiv_id>` if you have it.",
            [],
            self._meta(state),
        )

    # -----------------------------
    # Title matching helpers
    # -----------------------------
    _WS_RE = re.compile(r"\s+")
    _PUNCT_RE = re.compile(r"[^a-z0-9\s]")

    def _normalize_title(self, s: str) -> str:
        s = (s or "").strip().lower()
        s = self._WS_RE.sub(" ", s)
        s = self._PUNCT_RE.sub(" ", s)
        s = self._WS_RE.sub(" ", s).strip()
        return s

    def _find_best_title_match(
        self, state, query: str
    ) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
        qn = self._normalize_title(query)
        if not qn:
            return None, []

        hay = state.prefetched_results or []
        if not hay:
            return None, []

        # 1) Exact match
        for p in hay:
            tn = self._normalize_title(str(p.get("title") or ""))
            if tn and tn == qn:
                return p, []

        # 2) Substring match
        substring_matches: List[Dict[str, Any]] = []
        for p in hay:
            tn = self._normalize_title(str(p.get("title") or ""))
            if tn and qn in tn:
                substring_matches.append(p)
        if len(substring_matches) == 1:
            return substring_matches[0], []
        if len(substring_matches) > 1:
            return None, substring_matches

        # 3) Fuzzy match
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for p in hay:
            tn = self._normalize_title(str(p.get("title") or ""))
            if not tn:
                continue
            score = difflib.SequenceMatcher(a=qn, b=tn).ratio()
            scored.append((score, p))

        if not scored:
            return None, []

        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, best_paper = scored[0]

        if best_score >= 0.82:
            return best_paper, []

        candidates: List[Dict[str, Any]] = [p for s, p in scored[:5] if s >= 0.72]
        if candidates:
            return None, candidates

        return None, []

    async def _do_paper(
        self, state, arg: str
    ) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
        arxiv_id = (arg or "").strip()
        if not arxiv_id:
            return "Usage: `paper <arxiv_id>` (example: `paper 2003.10303`)", [], self._meta(state)

        try:
            paper = await self.arxiv.get_paper(arxiv_id)
            if not paper:
                return f"No paper found for arXiv id: {arxiv_id}", [], self._meta(state)

            paper_dict = self._paper_to_dict(paper)
            return self._format_paper_reply(paper_dict), [paper_dict], self._meta(state)

        except Exception:
            for p in state.prefetched_results:
                pid = str(p.get("arxiv_id") or p.get("id") or "")
                if pid == arxiv_id:
                    return self._format_paper_reply(p), [p], self._meta(state)
            return f"No paper found for arXiv id: {arxiv_id}", [], self._meta(state)

    # -----------------------------
    # Progressive prefetch (chunking)
    # -----------------------------
    async def _ensure_loaded(self, state, needed: int) -> None:
        if needed <= len(state.prefetched_results):
            return

        if len(state.prefetched_results) >= self.hard_total_cap:
            return

        if state.total_results is not None and len(state.prefetched_results) >= state.total_results:
            return

        while len(state.prefetched_results) < needed:
            start = len(state.prefetched_results)
            remaining_cap = self.hard_total_cap - start
            chunk = min(self.prefetch_chunk_size, remaining_cap)

            if state.total_results is not None:
                remaining_total = max(0, state.total_results - start)
                chunk = min(chunk, remaining_total)

            if chunk <= 0:
                return

            s = state.meta.get("search") or {}
            fetched, total = await self._fetch_from_backend(
                topic=(s.get("topic") or ""),
                author=(s.get("author") or ""),
                from_year=s.get("from_year"),
                to_year=s.get("to_year"),
                categories=s.get("categories") or [],
                start=start,
                max_results=chunk,
            )

            if state.total_results is None and total is not None:
                state.total_results = total

            if not fetched:
                return

            state.prefetched_results.extend(fetched)

            if len(fetched) < chunk:
                return

    def _paper_to_dict(self, p: Any) -> Dict[str, Any]:
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
        self,
        topic: str,
        author: str = "",
        from_year: Optional[int] = None,
        to_year: Optional[int] = None,
        categories: Optional[List[str]] = None,
        start: int = 0,
        max_results: int = 10,
    ) -> Tuple[List[Dict[str, Any]], Optional[int]]:
        print(
            f"[CHATBOT] backend.search(topic={topic!r}, author={author!r}, from_year={from_year!r}, to_year={to_year!r}, "
            f"categories={categories!r}, start={start}, max_results={max_results})"
        )
        resp = await self.arxiv.search(
            topic=topic,
            author=(author or "").strip() or None,
            from_year=from_year,
            to_year=to_year,
            categories=categories or None,
            start=start,
            max_results=max_results,
        )

        if resp is None:
            return [], None

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

    def _format_published_date(self, p: Dict[str, Any]) -> str:
        val = (
            p.get("published")
            or p.get("published_date")
            or p.get("created")
            or p.get("date")
            or ""
        )

        if isinstance(val, dict):
            val = val.get("date") or val.get("value") or ""

        if not isinstance(val, str):
            val = str(val) if val is not None else ""

        val = val.strip()
        if not val:
            return ""

        if "T" in val:
            return val.split("T", 1)[0].strip()

        return val[:10].strip()

    def _format_authors(self, p: Dict[str, Any], max_authors: int = 6) -> str:
        authors = p.get("authors") or p.get("author") or p.get("creators") or []
        if isinstance(authors, list):
            clean = [str(a).strip() for a in authors if str(a).strip()]
            if not clean:
                return ""
            if len(clean) <= max_authors:
                return ", ".join(clean)
            return ", ".join(clean[:max_authors]) + ", et al."
        if authors:
            return str(authors).strip()
        return ""

    def _format_list_reply(self, state, start: int, page: List[Dict[str, Any]], note: str = "") -> str:
        if not page:
            return "No results found."

        total_str = "?"
        if state.total_results is not None:
            total_str = str(state.total_results)

        shown_from = start + 1
        shown_to = start + len(page)
        loaded = len(state.prefetched_results)

        lines: List[str] = []
        lines.append(f"Showing {shown_from}-{shown_to} of {total_str} for **{state.current_query}** {note}".strip())
        lines.append(f"(loaded in session cache: {loaded})")

        for i, p in enumerate(page, start=1):
            title = (p.get("title") or "").strip() or "Untitled"
            authors_str = self._format_authors(p, max_authors=6)
            published = self._format_published_date(p)

            meta_bits: List[str] = []
            if authors_str:
                meta_bits.append(f"Authors: {authors_str}")
            if published:
                meta_bits.append(f"Published: {published}")

            if meta_bits:
                lines.append(f"{i}) {title}\n   " + " | ".join(meta_bits))
            else:
                lines.append(f"{i}) {title}")

        return "\n".join(lines)

    def _format_paper_reply(self, p: Dict[str, Any]) -> str:
        title = (p.get("title") or "").strip() or "Untitled"

        authors_str = self._format_authors(p, max_authors=8)
        published = self._format_published_date(p)

        summary = (p.get("summary") or p.get("abstract") or "").strip()
        if len(summary) > 900:
            summary = summary[:900].rstrip() + "..."

        lines: List[str] = [f"**{title}**"]
        if authors_str:
            lines.append(f"Authors: {authors_str}")
        if published:
            lines.append(f"Published: {published}")
        if summary:
            lines.append("")
            lines.append(summary)

        return "\n".join(lines)

    def _help_text(self) -> str:
        return (
            "Actions allowed:\n"
            "- `help`\n"
            "- `reset`\n"
            "- `search <topic>` (or just type a topic)\n"
            "- `search ... by <author>`\n"
            "- `search ... between <YYYY> and <YYYY>`\n"
            "- `search ... in <category>` (e.g., cs.AI, cs.LG, stat.ML)\n"
            "- `next`\n"
            "- `open <n>` (opens the nth paper on current page; if not on page, tries nth overall)\n"
            "- `open <title>` (opens the best matching paper title from current search)\n"
            "- `paper <arxiv_id>` (open by exact arXiv id)\n\n"
            "Examples:\n"
            "- ai in healthcare\n"
            "- papers by Andrew Ng\n"
            "- ai in healthcare between 2020 and 2022\n"
            "- cs.AI papers between 2020 and 2022\n"
        )

    
    def _describe_search(
        self,
        topic: str,
        author: str,
        from_year: Optional[int],
        to_year: Optional[int],
        categories: List[str],
    ) -> str:
        """
        Human-friendly description of the current structured search.

        Stored into SessionState.current_query so UX stays consistent with older
        "topic-only" searches (used in headers like: Showing X results for ...).
        """
        parts: List[str] = []
        if topic:
            parts.append(topic)
        if author:
            parts.append(f'by {author}')
        if from_year is not None and to_year is not None:
            parts.append(f'between {from_year} and {to_year}')
        elif from_year is not None:
            parts.append(f'from {from_year}')
        elif to_year is not None:
            parts.append(f'until {to_year}')
        if categories:
            parts.append("in " + ", ".join(categories))
        # Fallback (shouldn't happen because we validate at least one constraint)
        return " ".join(parts).strip() or "(search)"

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