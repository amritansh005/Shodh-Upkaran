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
    def __init__(
        self,
        arxiv: ArxivBackendClient,
        llm: LLMClient,
        store: InMemorySessionStore,
    ) -> None:
        self.arxiv = arxiv
        self.llm = llm
        self.store = store

    async def handle(
        self, session_id: str, message: str
    ) -> Tuple[str, Optional[List[Paper]], Dict[str, Any]]:
        state = await self.store.get(session_id)
        msg = (message or "").strip()

        if msg.lower() in {"help", "commands"}:
            return (
                "Commands:\n"
                "- Search anything: `ai in healthcare`\n"
                "- Or: `search: ai in healthcare`\n"
                "- `next`\n"
                "- `open 3`\n"
                "- `paper 2401.01234`\n"
                "- `reset`",
                None,
                {},
            )

        if msg.lower() in {"reset", "clear"}:
            await self.store.clear(session_id)
            return "Session cleared. Tell me a topic to search on arXiv.", None, {}

        if not self.llm.enabled():
            return (
                "Azure OpenAI is not configured. Add AZURE_* settings in chatbot_service/.env and restart.",
                None,
                {"need_llm": True},
            )

        intent = self._intent_from_llm(state, msg)
        action = intent.get("action", "search")

        # -----------------------
        # SEARCH
        # -----------------------
        if action == "search":
            query = (intent.get("query") or "").strip()
            start = int(intent.get("start", 0))
            max_results = int(intent.get("max_results", state.page_size))

            if not query:
                return "Tell me what topic to search (example: AI in healthcare).", None, {}

            try:
                data = await self.arxiv.search(topic=query, start=start, max_results=max_results)
            except httpx.HTTPStatusError as e:
                status = e.response.status_code if e.response is not None else None
                if status in (502, 503, 504):
                    return (
                        "arXiv is slow/unreachable right now. Please try again in a moment.",
                        None,
                        {"action": "search", "error": "arxiv_backend_upstream", "status": status},
                    )
                return (
                    f"Search failed (backend status {status}). Please try again.",
                    None,
                    {"action": "search", "error": "arxiv_backend_http_error", "status": status},
                )
            except httpx.RequestError:
                return (
                    "Could not reach the arXiv backend. Is arxiv_backend running on http://127.0.0.1:8000 ?",
                    None,
                    {"action": "search", "error": "arxiv_backend_unreachable"},
                )
            except Exception:
                return (
                    "Something went wrong while searching. Try again.",
                    None,
                    {"action": "search", "error": "unknown"},
                )

            papers = data["papers"]

            state.last_query = query
            state.last_start = start
            state.page_size = max_results
            state.last_results = papers

            reply = self._format_search_reply(query, start, papers)
            return reply, papers, {"action": "search", "start": start}

        # -----------------------
        # NEXT PAGE
        # -----------------------
        if action == "next":
            if not state.last_query:
                return "No active search yet. Ask me a topic first (example: AI in healthcare).", None, {}

            start = state.last_start + state.page_size

            try:
                data = await self.arxiv.search(topic=state.last_query, start=start, max_results=state.page_size)
            except httpx.HTTPStatusError as e:
                status = e.response.status_code if e.response is not None else None
                if status in (502, 503, 504):
                    return (
                        "arXiv is slow/unreachable right now. Please try `next` again in a moment.",
                        None,
                        {"action": "next", "error": "arxiv_backend_upstream", "status": status},
                    )
                return (
                    f"Next page failed (backend status {status}). Please try again.",
                    None,
                    {"action": "next", "error": "arxiv_backend_http_error", "status": status},
                )
            except httpx.RequestError:
                return (
                    "Could not reach the arXiv backend. Is arxiv_backend running on http://127.0.0.1:8000 ?",
                    None,
                    {"action": "next", "error": "arxiv_backend_unreachable"},
                )
            except Exception:
                return (
                    "Something went wrong while getting the next page. Try again.",
                    None,
                    {"action": "next", "error": "unknown"},
                )

            papers = data["papers"]

            state.last_start = start
            state.last_results = papers

            reply = self._format_search_reply(state.last_query, start, papers)
            return reply, papers, {"action": "next", "start": start}

        # -----------------------
        # OPEN Nth PAPER (from last search results)
        # -----------------------
        if action == "open":
            index = int(intent.get("index", 1))
            if not state.last_results:
                return "No results stored yet. Search a topic first.", None, {}

            if index < 1 or index > len(state.last_results):
                return f"Pick a number between 1 and {len(state.last_results)}.", None, {}

            chosen = state.last_results[index - 1]

            try:
                full = await self.arxiv.get_paper(chosen.arxiv_id)
            except httpx.HTTPStatusError as e:
                status = e.response.status_code if e.response is not None else None
                if status in (502, 503, 504):
                    return (
                        "arXiv is slow/unreachable right now. Please try opening that paper again.",
                        None,
                        {"action": "open", "error": "arxiv_backend_upstream", "status": status, "index": index},
                    )
                return (
                    f"Couldn’t fetch that paper (backend status {status}). Try again.",
                    None,
                    {"action": "open", "error": "arxiv_backend_http_error", "status": status, "index": index},
                )
            except httpx.RequestError:
                return (
                    "Could not reach the arXiv backend. Is arxiv_backend running on http://127.0.0.1:8000 ?",
                    None,
                    {"action": "open", "error": "arxiv_backend_unreachable", "index": index},
                )
            except Exception:
                return (
                    "Couldn’t fetch that paper right now. Try again.",
                    None,
                    {"action": "open", "error": "unknown", "index": index},
                )

            if not full:
                return "Couldn’t fetch that paper right now.", None, {"action": "open", "index": index}

            reply = self._format_paper(full)
            return reply, [full], {"action": "open", "index": index}

        # -----------------------
        # PAPER BY ID
        # -----------------------
        if action == "paper":
            arxiv_id = (intent.get("arxiv_id") or "").strip()
            if not arxiv_id:
                return "Send: paper <arxiv_id> (example: paper 2401.01234)", None, {}

            try:
                full = await self.arxiv.get_paper(arxiv_id)
            except httpx.HTTPStatusError as e:
                status = e.response.status_code if e.response is not None else None
                if status == 404:
                    return "Paper not found.", None, {"action": "paper", "arxiv_id": arxiv_id, "status": status}
                if status in (502, 503, 504):
                    return (
                        "arXiv is slow/unreachable right now. Please try again in a moment.",
                        None,
                        {"action": "paper", "error": "arxiv_backend_upstream", "status": status, "arxiv_id": arxiv_id},
                    )
                return (
                    f"Couldn’t fetch that paper (backend status {status}). Try again.",
                    None,
                    {"action": "paper", "error": "arxiv_backend_http_error", "status": status, "arxiv_id": arxiv_id},
                )
            except httpx.RequestError:
                return (
                    "Could not reach the arXiv backend. Is arxiv_backend running on http://127.0.0.1:8000 ?",
                    None,
                    {"action": "paper", "error": "arxiv_backend_unreachable", "arxiv_id": arxiv_id},
                )
            except Exception:
                return (
                    "Couldn’t fetch that paper right now. Try again.",
                    None,
                    {"action": "paper", "error": "unknown", "arxiv_id": arxiv_id},
                )

            if not full:
                return "Paper not found.", None, {"action": "paper", "arxiv_id": arxiv_id}

            reply = self._format_paper(full)
            return reply, [full], {"action": "paper", "arxiv_id": arxiv_id}

        return "Tell me a topic to search, or type `help`.", None, {"action": "fallback"}

    # -----------------------
    # LLM routing
    # -----------------------
    def _intent_from_llm(self, state: SessionState, msg: str) -> Dict[str, Any]:
        """
        Returns STRICT JSON only:
        - search: {action:"search", query:"...", start:0, max_results:10}
        - next:   {action:"next"}
        - open:   {action:"open", index:3}
        - paper:  {action:"paper", arxiv_id:"2401.01234"}
        """
        system = (
            "You are an intent router for an arXiv chatbot.\n"
            "Return ONLY valid JSON. No markdown.\n\n"
            "Actions:\n"
            "1) search: user wants papers on a topic\n"
            "2) next: next page of last search\n"
            "3) open: open Nth paper from last results\n"
            "4) paper: fetch paper by explicit arXiv id\n\n"
            "Rules:\n"
            "- If user says 'next' or 'next page' => {\"action\":\"next\"}\n"
            "- If user says 'open 3' => {\"action\":\"open\",\"index\":3}\n"
            "- If user says 'paper 2401.01234' or message looks like an arXiv id => action=paper\n"
            "- Otherwise treat message as a search query: action=search with query=message\n"
            "- Default start=0 and max_results=10 for searches.\n"
        )

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": msg},
        ]

        raw = self.llm.chat(messages, temperature=0.0).strip()

        try:
            data = json.loads(raw)
            if isinstance(data, dict) and "action" in data:
                # normalize minimal fields
                if data["action"] == "search":
                    data.setdefault("start", 0)
                    data.setdefault("max_results", 10)
                    # Some models may forget to include "query" even for search.
                    data.setdefault("query", msg)
                return data
        except Exception:
            pass

        # safe fallback
        return {"action": "search", "query": msg, "start": 0, "max_results": 10}

    # -----------------------
    # Formatting helpers
    # -----------------------
    def _format_search_reply(self, query: str, start: int, papers: List[Paper]) -> str:
        if not papers:
            return f"No results for **{query}**. Try a different phrase."

        lines = [f"Showing {len(papers)} results for **{query}** (start={start}):"]
        for i, p in enumerate(papers, start=1):
            lines.append(f"{i}) {p.title} — {p.arxiv_id}")
        lines.append("\nSay `open 1` (or any number), or `next`.")
        return "\n".join(lines)

    def _format_paper(self, p: Paper) -> str:
        authors = ", ".join(p.authors) if p.authors else "Unknown"
        pdf = p.pdf_url or "N/A"
        abstract = (p.abstract or "").strip()
        return (
            f"**{p.title}**\n"
            f"arXiv: {p.arxiv_id}\n"
            f"Authors: {authors}\n\n"
            f"{abstract}\n\n"
            f"PDF: {pdf}\n\n"
            f"(Later: download → vector DB → chat-with-PDF will be handled by arxiv_backend.)"
        )