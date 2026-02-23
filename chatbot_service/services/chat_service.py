# chatbot_service/services/chat_service.py
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import httpx

from chatbot_service.schemas import Paper
from chatbot_service.services.arxiv_backend_client import ArxivBackendClient
from chatbot_service.services.llm_client import LLMClient
from chatbot_service.services.session_store import InMemorySessionStore, SessionState


class ChatService:
    """
    Chatbot brain:
    - Uses LLM ONLY to route intent (search/next/open/paper/help/reset)
    - Uses arxiv_backend via ArxivBackendClient for data
    - Uses InMemorySessionStore for conversation state

    IMPORTANT:
    Your current InMemorySessionStore does NOT implement .set().
    That is OK: store.get() returns the *same mutable SessionState object*
    stored inside the dict, so mutating `state` is enough.
    """

    def __init__(self, arxiv: ArxivBackendClient, llm: LLMClient, store: InMemorySessionStore):
        self.arxiv = arxiv
        self.llm = llm
        self.store = store

    async def handle_message(self, session_id: str, message: str) -> Tuple[str, Optional[List[Paper]], Dict[str, Any]]:
        state = await self.store.get(session_id)
        msg = (message or "").strip()

        if not self.llm.enabled():
            return (
                "Azure OpenAI is not configured. Add AZURE_* settings in chatbot_service/.env and restart.",
                None,
                {"error": "llm_not_configured"},
            )

        # LLM is the ONLY router.
        intent = self._intent_from_llm(state, msg)
        action = (intent.get("action") or "search").strip()

        # -----------------------
        # HELP
        # -----------------------
        if action == "help":
            return (
                "Commands:\n"
                "- Search: type any topic (example: `ai in healthcare`)\n"
                "- Next page: `next`\n"
                "- Open from list: `open 3` / `open the 9th paper`\n"
                "- Fetch by id: `paper 2401.01234`\n"
                "- Reset: `reset`",
                None,
                {"action": "help"},
            )

        # -----------------------
        # RESET
        # -----------------------
        if action == "reset":
            await self.store.clear(session_id)
            return "Session cleared. Tell me a topic to search on arXiv.", None, {"action": "reset"}

        # -----------------------
        # SEARCH
        # -----------------------
        if action == "search":
            query = (intent.get("query") or msg).strip()
            start = self._safe_int(intent.get("start"), default=0)
            max_results = self._safe_int(intent.get("max_results"), default=state.page_size)

            # Guardrails
            if max_results <= 0:
                max_results = state.page_size or 10
            if max_results > 50:
                max_results = 50  # keep responses sane

            papers = await self._search(query=query, start=start, max_results=max_results)

            # Save state for follow-ups like "next", "open 9"
            state.last_query = query
            state.last_start = start
            state.page_size = max_results
            state.last_results = papers

            text = self._format_search_results(query, start, papers)
            return text, papers, {"action": "search", "query": query, "start": start, "max_results": max_results}

        # -----------------------
        # NEXT PAGE
        # -----------------------
        if action == "next":
            if not state.last_query:
                return (
                    "I don’t have an active search yet. Tell me a topic to search on arXiv.",
                    None,
                    {"action": "next", "error": "no_last_query"},
                )

            next_start = state.last_start + (state.page_size or 10)
            papers = await self._search(query=state.last_query, start=next_start, max_results=state.page_size or 10)

            state.last_start = next_start
            state.last_results = papers

            text = self._format_search_results(state.last_query, next_start, papers)
            return text, papers, {"action": "next", "query": state.last_query, "start": next_start, "max_results": state.page_size}

        # -----------------------
        # OPEN Nth PAPER
        # -----------------------
        if action == "open":
            index = self._safe_int(intent.get("index"), default=-1)
            if index <= 0:
                return "Tell me which paper number to open (example: `open 2`).", None, {"action": "open", "error": "invalid_index"}

            if not state.last_results:
                # LLM was instructed to return help instead, but keep this safe guard.
                return "I don’t have a list yet. Search a topic first, then say `open 3`.", None, {"action": "open", "error": "no_last_results"}

            if index > len(state.last_results):
                return (
                    f"I only have {len(state.last_results)} papers in the current list. "
                    f"Try `open 1` to `open {len(state.last_results)}`.",
                    None,
                    {"action": "open", "error": "index_out_of_range", "count": len(state.last_results)},
                )

            paper = state.last_results[index - 1]
            text = self._format_paper_detail(paper)
            return text, [paper], {"action": "open", "index": index, "arxiv_id": paper.arxiv_id}

        # -----------------------
        # PAPER BY ID
        # -----------------------
        if action == "paper":
            arxiv_id = (intent.get("arxiv_id") or "").strip()
            if not arxiv_id:
                return "Tell me the arXiv id (example: `paper 2401.01234`).", None, {"action": "paper", "error": "missing_arxiv_id"}

            paper = await self._get_paper(arxiv_id)
            if not paper:
                return f"I couldn’t find that paper: {arxiv_id}", None, {"action": "paper", "error": "not_found", "arxiv_id": arxiv_id}

            text = self._format_paper_detail(paper)
            return text, [paper], {"action": "paper", "arxiv_id": arxiv_id}

        # -----------------------
        # Unknown action => treat as search (LLM-proof)
        # -----------------------
        papers = await self._search(query=msg, start=0, max_results=state.page_size or 10)
        state.last_query = msg
        state.last_start = 0
        state.last_results = papers

        text = self._format_search_results(msg, 0, papers)
        return text, papers, {"action": "search_fallback", "query": msg, "start": 0, "max_results": state.page_size or 10}

    async def _search(self, query: str, start: int, max_results: int) -> List[Paper]:
        """
        Call the arXiv backend client and return a List[Paper].

        Your current ArxivBackendClient signature:
            search(topic: str, start: int, max_results: int, ...)-> Dict[str,Any]

        Older iterations might have used q= or query=.
        We try safely in order.
        """
        try:
            try:
                resp = await self.arxiv.search(topic=query, start=start, max_results=max_results)
            except TypeError:
                # Backward-compat (if you swap client versions later)
                try:
                    resp = await self.arxiv.search(q=query, start=start, max_results=max_results)  # type: ignore[misc]
                except TypeError:
                    resp = await self.arxiv.search(query=query, start=start, max_results=max_results)  # type: ignore[misc]
        except httpx.HTTPError:
            return []

        # Current client returns dict: {"papers":[Paper,...], "total_results":..., "error":...}
        if isinstance(resp, dict):
            if resp.get("error"):
                return []
            papers = resp.get("papers") or resp.get("results") or []
            return papers if isinstance(papers, list) else []

        # Some older client versions might return List[Paper]
        return resp  # type: ignore[return-value]

    async def _get_paper(self, arxiv_id: str) -> Optional[Paper]:
        """
        Current client method is get_paper(arxiv_id=...).
        Keep a fallback for older client versions that used paper(...).
        """
        try:
            if hasattr(self.arxiv, "get_paper"):
                return await self.arxiv.get_paper(arxiv_id=arxiv_id)  # type: ignore[attr-defined]
            return await self.arxiv.paper(arxiv_id=arxiv_id)  # type: ignore[attr-defined]
        except httpx.HTTPError:
            return None

    # -----------------------
    # LLM routing
    # -----------------------
    def _intent_from_llm(self, state: SessionState, msg: str) -> Dict[str, Any]:
        """
        Returns STRICT JSON only (no markdown, no extra keys):

        - help:  {"action":"help"}
        - reset: {"action":"reset"}
        - search:{"action":"search","query":"...","start":0,"max_results":10}
        - next:  {"action":"next"}
        - open:  {"action":"open","index":3}
        - paper: {"action":"paper","arxiv_id":"2401.01234"}
        """

        # Context for references like “open the 9th one”
        last_results = state.last_results or []
        preview_lines: List[str] = []
        for i, p in enumerate(last_results[:25], start=1):
            preview_lines.append(f"{i}) {p.title} — {p.arxiv_id}")
        preview = "\n".join(preview_lines) if preview_lines else "(none)"

        system = (
            "You are an intent router for an arXiv chatbot.\n"
            "Return ONLY a single JSON object. No markdown. No commentary.\n"
            "Use EXACTLY one of these actions: help, reset, search, next, open, paper.\n\n"
            "State (may be empty):\n"
            f"- last_query: {state.last_query!r}\n"
            f"- last_start: {state.last_start}\n"
            f"- page_size: {state.page_size}\n"
            f"- last_results_count: {len(last_results)}\n"
            "- last_results_preview (1-indexed):\n"
            f"{preview}\n\n"
            "Output JSON schema (no extra keys):\n"
            "- help  => {\"action\":\"help\"}\n"
            "- reset => {\"action\":\"reset\"}\n"
            "- next  => {\"action\":\"next\"}\n"
            "- open  => {\"action\":\"open\",\"index\":<int>}\n"
            "- paper => {\"action\":\"paper\",\"arxiv_id\":\"<id>\"}\n"
            "- search=> {\"action\":\"search\",\"query\":\"<topic>\",\"start\":<int>,\"max_results\":<int>}\n\n"
            "Routing rules:\n"
            "1) If user asks for help/commands/examples => action=help.\n"
            "2) If user asks to reset/clear/start over => action=reset.\n"
            "3) If user asks for next page / more results / continue => action=next.\n"
            "4) If user asks to open/show/display a specific item from the last results by number or ordinal\n"
            "   (e.g., 'open 9', 'open the ninth paper', 'show me the 3rd one') => action=open with that index.\n"
            "5) If user provides an explicit arXiv id (e.g., 2401.01234 or 2103.14954) or says 'paper <id>' => action=paper.\n"
            "6) Otherwise treat the message as a search request => action=search with query=the message.\n\n"
            "Defaults for search: start=0, max_results=10 unless user explicitly requests another page size.\n"
            "If last_results_count is 0 and user asks to open, return action=help instead of guessing.\n"
        )

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": msg},
        ]

        raw = self.llm.chat(messages, temperature=0.0).strip()

        data = self._safe_json_object(raw)
        if isinstance(data, dict) and "action" in data:
            if data["action"] == "search":
                data.setdefault("start", 0)
                data.setdefault("max_results", 10)
                data.setdefault("query", msg)
            return data

        # Safe fallback if parsing fails
        return {"action": "search", "query": msg, "start": 0, "max_results": 10}

    def _safe_json_object(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Best-effort JSON object parser without regex.
        Extracts the first top-level JSON object by brace scanning.
        """
        if not text:
            return None

        text = text.strip()

        # Fast path: whole string is JSON
        try:
            obj = json.loads(text)
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass

        start = text.find("{")
        if start == -1:
            return None

        depth = 0
        in_str = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
                continue

            if ch == '"':
                in_str = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    try:
                        obj = json.loads(candidate)
                        return obj if isinstance(obj, dict) else None
                    except Exception:
                        return None

        return None

    # -----------------------
    # Formatting helpers
    # -----------------------
    def _format_search_results(self, query: str, start: int, papers: List[Paper]) -> str:
        if not papers:
            return f"No results found for **{query}**."

        lines = [f"Showing {len(papers)} results for **{query}** (start={start}):"]
        for i, p in enumerate(papers, start=1):
            lines.append(f"{i}) {p.title} — {p.arxiv_id}")
        return "\n".join(lines)

    def _format_paper_detail(self, p: Paper) -> str:
        authors = ", ".join(p.authors or [])
        abstract = (p.abstract or "").strip()
        return f"**{p.title}**\narXiv: {p.arxiv_id}\nAuthors: {authors}\n\n{abstract}"

    @staticmethod
    def _safe_int(value: Any, default: int) -> int:
        try:
            return int(value)
        except Exception:
            return default