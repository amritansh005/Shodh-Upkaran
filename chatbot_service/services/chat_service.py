from __future__ import annotations

import difflib
import re
from typing import Any, Dict, List, Optional, Tuple

import asyncio


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
    # Pending "Is it a new search?" confirmation handling
    # (Small additions; does NOT disrupt existing logic)
    # -----------------------------
    _YES_RE = re.compile(
        r"^(?:y|yes|yeah|yep|ok|okay|sure|do it|go ahead|new search|yes its a new search|yes it's a new search)\b",
        re.IGNORECASE,
    )
    _NO_RE = re.compile(r"^(?:n|no|nope|nah)\b", re.IGNORECASE)
    _CANCEL_RE = re.compile(r"^(?:forget it|never mind|nevermind|cancel|drop it|ignore that)\b", re.IGNORECASE)

    def _is_yes(self, msg: str) -> bool:
        return bool(self._YES_RE.match((msg or "").strip()))

    def _is_no(self, msg: str) -> bool:
        return bool(self._NO_RE.match((msg or "").strip()))

    def _is_cancel(self, msg: str) -> bool:
        return bool(self._CANCEL_RE.match((msg or "").strip()))

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

        # -----------------------------
        # NEW: Handle pending "new search?" confirmation BEFORE calling LLM
        # -----------------------------
        pending = (state.meta or {}).get("pending_search")
        if pending:
            if self._is_yes(msg):
                # Confirmed -> directly run the stored search
                state.meta.pop("pending_search", None)

                topic = str(pending.get("topic") or "").strip()
                author = str(pending.get("author") or "").strip()
                from_year = pending.get("from_year", None)
                to_year = pending.get("to_year", None)
                categories = pending.get("categories") or []

                print(f"[CHATBOT] pending_search confirmed -> topic={topic!r}, author={author!r}, years={from_year}-{to_year}, cats={categories!r}")

                return await self._do_search(
                    state,
                    topic=topic,
                    author=author,
                    from_year=from_year,
                    to_year=to_year,
                    categories=categories,
                )

            if self._is_no(msg) or self._is_cancel(msg):
                # Denied/canceled -> stay in current results and guide open-by-number/title
                state.meta.pop("pending_search", None)
                print("[CHATBOT] pending_search denied/cancelled -> staying in current results")

                return (
                    "Okay — staying with the current results. "
                    "Please specify which paper to open by serial number (example: `open 7`) or paste the title.",
                    [],
                    self._meta(state),
                )

            # Neither approval nor denial: user changed request (e.g. "Forget it. Search ai in healthcare")
            # Clear pending and proceed normally.
            state.meta.pop("pending_search", None)
            print("[CHATBOT] pending_search cleared due to new user request (not yes/no)")

        # LLM routing + slot extraction
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

        # -----------------------------
        # Session-first resolution (confidence thresholds)
        # If a list is on screen (active results context), interpret OPEN/READ requests
        # as selecting from that list first. Only run a new SEARCH if the current list
        # has zero reasonable candidates for what the user asked.
        # -----------------------------
        has_arc = bool(state.current_query) and bool(state.last_results or state.prefetched_results)
        looks_like_open = self._looks_like_open_request(msg)

        if has_arc and action == "search" and looks_like_open:
            # LLM may misclassify "open the paper by X" as a SEARCH (author search).
            # Prefer opening from the current list first.
            return await self._do_open_from_context(state, intent, user_message=msg)

        if has_arc and action == "open":
            # If OPEN action lacks index/title/arxiv_id but includes selectors, resolve against current list.
            if index is None and not title and not arxiv_id and (
                author or topic or (from_year is not None) or (to_year is not None) or categories
            ):
                return await self._do_open_from_context(state, intent, user_message=msg)

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

            # If no title/index but user wrote an open-like message, try context resolution
            if has_arc and looks_like_open:
                return await self._do_open_from_context(state, intent, user_message=msg)

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
            categories=categories or [],
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
    # Session-first OPEN resolution (no new search unless needed)
    # -----------------------------
    _OPEN_VERBS = (
        "open",
        "read",
        "show me",
        "show",
        "details",
        "summarize",
        "summary",
        "explain",
        "tell me about",
    )

    def _looks_like_open_request(self, msg: str) -> bool:
        m = (msg or "").strip().lower()
        if not m:
            return False
        # Require an open/read/details cue so normal "papers by X" stays SEARCH.
        return any(v in m for v in self._OPEN_VERBS)

    def _wants_show_all(self, msg: str) -> bool:
        m = (msg or "").lower()
        return ("show all" in m) or ("list all" in m) or ("all matches" in m)

    def _wants_choose_one(self, msg: str) -> bool:
        m = (msg or "").lower()
        return ("choose one" in m) or ("pick one" in m) or ("you choose" in m) or ("just choose" in m)

    def _paper_year(self, p: Dict[str, Any]) -> Optional[int]:
        d = self._format_published_date(p)
        if d and len(d) >= 4 and d[:4].isdigit():
            try:
                return int(d[:4])
            except Exception:
                return None
        return None

    def _author_match(self, p: Dict[str, Any], author_q: str) -> bool:
        aq = (author_q or "").strip().lower()
        if not aq:
            return True
        auth = p.get("authors") or p.get("author") or ""
        if isinstance(auth, list):
            blob = " ".join([str(x) for x in auth])
        else:
            blob = str(auth)
        return aq in blob.lower()

    def _topic_match(self, p: Dict[str, Any], topic_q: str) -> bool:
        tq = (topic_q or "").strip().lower()
        if not tq:
            return True
        title = str(p.get("title") or "").lower()
        summ = str(p.get("summary") or p.get("abstract") or "").lower()
        return (tq in title) or (tq in summ)

    def _year_match(self, p: Dict[str, Any], from_year: Optional[int], to_year: Optional[int]) -> bool:
        if from_year is None and to_year is None:
            return True
        y = self._paper_year(p)
        if y is None:
            return False
        if from_year is not None and y < int(from_year):
            return False
        if to_year is not None and y > int(to_year):
            return False
        return True

    def _filter_candidates(
        self,
        papers: List[Dict[str, Any]],
        *,
        author: str,
        topic: str,
        from_year: Optional[int],
        to_year: Optional[int],
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for p in papers:
            if not self._author_match(p, author):
                continue
            if not self._year_match(p, from_year, to_year):
                continue
            if not self._topic_match(p, topic):
                continue
            out.append(p)
        return out

    def _title_best_in(
        self, papers: List[Dict[str, Any]], query: str
    ) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
        qn = self._normalize_title(query)
        if not qn:
            return None, []

        # 1) Exact
        for p in papers:
            tn = self._normalize_title(str(p.get("title") or ""))
            if tn and tn == qn:
                return p, []

        # 2) Substring
        subs: List[Dict[str, Any]] = []
        for p in papers:
            tn = self._normalize_title(str(p.get("title") or ""))
            if tn and qn in tn:
                subs.append(p)
        if len(subs) == 1:
            return subs[0], []
        if len(subs) > 1:
            return None, subs

        # 3) Fuzzy
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for p in papers:
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

        cands = [p for s, p in scored[:10] if s >= 0.72]
        if cands:
            return None, cands
        return None, []

    def _indices_for_candidates(
        self, state, candidates: List[Dict[str, Any]]
    ) -> List[Tuple[int, Dict[str, Any]]]:
        # Prefer page indices if candidate is in last_results; else global indices.
        out: List[Tuple[int, Dict[str, Any]]] = []
        last = state.last_results or []
        # map arxiv_id to index on current page
        page_map: Dict[str, int] = {}
        for i, p in enumerate(last):
            pid = str(p.get("arxiv_id") or p.get("id") or "")
            if pid:
                page_map[pid] = i + 1

        for p in candidates:
            pid = str(p.get("arxiv_id") or p.get("id") or "")
            if pid and pid in page_map:
                out.append((page_map[pid], p))
                continue
            # fallback to loaded index
            try:
                gi = state.prefetched_results.index(p)
                out.append((gi + 1, p))
            except Exception:
                out.append((0, p))
        return out

    def _format_candidate_choices(
        self,
        state,
        candidates: List[Dict[str, Any]],
        header: str,
        limit: int = 10,
    ) -> str:
        pairs = self._indices_for_candidates(state, candidates)
        lines = [header]
        for n, (idx, p) in enumerate(pairs):
            if n >= limit:
                break
            t = str(p.get("title") or "").strip()
            a_str = self._format_authors(p, max_authors=6)
            d = self._format_published_date(p)
            prefix = f"{idx}) " if idx else "- "
            meta_bits: List[str] = []
            if a_str:
                meta_bits.append(f"Authors: {a_str}")
            if d:
                meta_bits.append(f"Published: {d}")
            if meta_bits:
                lines.append(f"{prefix}{t}\n   " + " | ".join(meta_bits))
            else:
                lines.append(f"{prefix}{t}")
        return "\n".join(lines)

    async def _do_open_from_context(
        self,
        state,
        intent: Dict[str, Any],
        user_message: str,
    ) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
        """Resolve an OPEN-like request using the current list first (confidence thresholds).

        - Try within current page (state.last_results) first.
        - Then try within all loaded results (state.prefetched_results).
        - Only if zero reasonable candidates exist, ask whether it is a new search.
        """
        author = str(intent.get("author") or "").strip()
        topic = str(intent.get("topic") or "").strip()
        title = str(intent.get("title") or "").strip()
        from_year = intent.get("from_year", None)
        to_year = intent.get("to_year", None)
        arxiv_id = str(intent.get("arxiv_id") or "").strip()
        index = intent.get("index", None)

        # If user provided an index or arXiv id, defer to explicit handlers.
        if index is not None:
            try:
                return await self._do_open_index(state, int(index))
            except Exception:
                pass
        if arxiv_id:
            return await self._do_paper(state, arxiv_id)

        page_view = state.last_results or []
        loaded = state.prefetched_results or []

        # Title-based selection (page-first, then loaded)
        if title:
            paper, cands = self._title_best_in(page_view, title)
            if paper is not None:
                return self._format_paper_reply(paper), [paper], self._meta(state)

            if cands:
                if self._wants_show_all(user_message):
                    txt = self._format_candidate_choices(
                        state,
                        cands,
                        header="Matches in the current page (reply `open <number>`):",
                        limit=15,
                    )
                    return txt, cands[:15], self._meta(state)

                if self._wants_choose_one(user_message):
                    chosen = cands[0]
                    return self._format_paper_reply(chosen), [chosen], self._meta(state)

                txt = self._format_candidate_choices(
                    state,
                    cands,
                    header="I found multiple close matches in the current page. Which one should I open? Reply `open <number>`:",
                    limit=10,
                )
                return txt, cands[:10], self._meta(state)

            paper2, cands2 = self._title_best_in(loaded, title)
            if paper2 is not None:
                return self._format_paper_reply(paper2), [paper2], self._meta(state)

            if cands2:
                if self._wants_show_all(user_message):
                    txt = self._format_candidate_choices(
                        state,
                        cands2,
                        header="Matches in the current loaded results (reply `open <number>`):",
                        limit=20,
                    )
                    return txt, cands2[:20], self._meta(state)

                if self._wants_choose_one(user_message):
                    chosen = cands2[0]
                    return self._format_paper_reply(chosen), [chosen], self._meta(state)

                txt = self._format_candidate_choices(
                    state,
                    cands2,
                    header="I found multiple close matches in the current results. Which one should I open? Reply `open <number>`:",
                    limit=10,
                )
                return txt, cands2[:10], self._meta(state)

            return (
                "I can't find that title in the current results. If you want a new search, say: `search <title>`.",
                [],
                self._meta(state),
            )

        # Non-title selectors (author/topic/year) — filter page first
        page_matches = self._filter_candidates(
            page_view,
            author=author,
            topic=topic,
            from_year=from_year,
            to_year=to_year,
        )

        if len(page_matches) == 1:
            p = page_matches[0]
            return self._format_paper_reply(p), [p], self._meta(state)

        if len(page_matches) > 1:
            if self._wants_show_all(user_message):
                txt = self._format_candidate_choices(
                    state,
                    page_matches,
                    header="Multiple matches in the current page (reply `open <number>`):",
                    limit=20,
                )
                return txt, page_matches[:20], self._meta(state)

            if self._wants_choose_one(user_message):
                chosen = page_matches[0]
                return self._format_paper_reply(chosen), [chosen], self._meta(state)

            txt = self._format_candidate_choices(
                state,
                page_matches,
                header="I found multiple matches in the current page. Which one should I open? Reply `open <number>`:",
                limit=10,
            )
            return txt, page_matches[:10], self._meta(state)

        # Try within loaded results
        loaded_matches = self._filter_candidates(
            loaded,
            author=author,
            topic=topic,
            from_year=from_year,
            to_year=to_year,
        )

        if len(loaded_matches) == 1:
            p = loaded_matches[0]
            return self._format_paper_reply(p), [p], self._meta(state)

        if len(loaded_matches) > 1:
            if self._wants_show_all(user_message):
                txt = self._format_candidate_choices(
                    state,
                    loaded_matches,
                    header="Multiple matches in your current results (reply `open <number>`):",
                    limit=30,
                )
                return txt, loaded_matches[:30], self._meta(state)

            if self._wants_choose_one(user_message):
                chosen = loaded_matches[0]
                return self._format_paper_reply(chosen), [chosen], self._meta(state)

            txt = self._format_candidate_choices(
                state,
                loaded_matches,
                header="I found multiple matches in your current results. Which one should I open? Reply `open <number>`:",
                limit=10,
            )
            return txt, loaded_matches[:10], self._meta(state)

        # Zero reasonable candidates in current list -> ask if it's a new search, and store pending search
        hint_parts: List[str] = []
        if author:
            hint_parts.append(f"by **{author}**")
        if topic:
            hint_parts.append(f"about **{topic}**")
        if from_year is not None and to_year is not None:
            hint_parts.append(f"between **{from_year}** and **{to_year}**")
        elif from_year is not None:
            hint_parts.append(f"from **{from_year}**")
        elif to_year is not None:
            hint_parts.append(f"until **{to_year}**")

        hint = " ".join(hint_parts).strip() or "that"

        state.meta["pending_search"] = {
            "topic": topic,
            "author": author,
            "from_year": from_year,
            "to_year": to_year,
            "categories": intent.get("categories") or [],
        }

        print(f"[CHATBOT] pending_search set -> {state.meta['pending_search']}")

        return (
            f"I don't see {hint} in the current results. Is it a new search?\n"
            f"- If **yes**, reply `yes` and I'll search for it.\n"
            f"- If **no**, tell me which paper to open (serial number like `open 7` or paste the title).",
            [],
            self._meta(state),
        )

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

        try:
            resp = await asyncio.wait_for(
                self.arxiv.search(
                    topic=topic,
                    author=(author or "").strip() or None,
                    from_year=from_year,
                    to_year=to_year,
                    categories=categories or None,
                    start=start,
                    max_results=max_results,
                ),
                timeout=90.0,
            )
        except asyncio.TimeoutError:
            return [], None

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

    def _format_authors(self, p: Dict[str, Any], max_authors: int = 6, prioritize_author: str = "") -> str:
        authors = p.get("authors") or p.get("author") or p.get("creators") or []
        if isinstance(authors, list):
            clean = [str(a).strip() for a in authors if str(a).strip()]
            if not clean:
                return ""

            # If this was an author-filtered search, prioritize showing that author
            # in the limited "first N authors + et al." preview.
            pa = (prioritize_author or "").strip().lower()
            if pa:
                best_i: Optional[int] = None
                best_score = 0  # 2=exact match, 1=substring match

                for i, name in enumerate(clean):
                    n = name.lower()
                    if n == pa:
                        best_i = i
                        best_score = 2
                        break
                    if pa in n and best_score < 1:
                        best_i = i
                        best_score = 1

                # Move matched author to front for display only
                if best_i is not None and best_i != 0:
                    matched = clean[best_i]
                    rest = [x for j, x in enumerate(clean) if j != best_i]
                    clean = [matched] + rest

            shown = clean[:max_authors]
            suffix = ", et al." if len(clean) > max_authors else ""
            return ", ".join(shown) + suffix

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
            searched_author = str((state.meta.get("search") or {}).get("author") or "").strip()
            authors_str = self._format_authors(p, max_authors=6, prioritize_author=searched_author)
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
            parts.append(f"by {author}")
        if from_year is not None and to_year is not None:
            parts.append(f"between {from_year} and {to_year}")
        elif from_year is not None:
            parts.append(f"from {from_year}")
        elif to_year is not None:
            parts.append(f"until {to_year}")
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
            # NEW: small meta signal for the LLM (does not disrupt anything)
            "pending_search": bool((state.meta or {}).get("pending_search")),
        }