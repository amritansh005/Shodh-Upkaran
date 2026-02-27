from __future__ import annotations

import json
import re
from typing import List, Dict, Any, Optional

from openai import AzureOpenAI


_INTENT_SYSTEM_PROMPT = """
You are an intent and slot extractor for an arXiv research assistant chatbot.

Return ONLY valid JSON. No markdown. No explanation. No extra text.

Schema (ALWAYS include all fields):

{
  "action": "search" | "summary" | "open" | "next" | "help" | "reset" | "paper" | "chat",
  "topic": string,
  "author": string,
  "from_year": integer or null,
  "to_year": integer or null,
  "categories": array of strings,
  "index": integer or null,
  "title": string,
  "arxiv_id": string,
  "chat_response": string
}

Core actions:
- search: user wants a NEW list of papers (topic/author/year/categories constraints)
- summary: user wants details/abstract/summary/data-related info about ONE specific paper
- open: user wants the PDF/download link for ONE specific paper
- next: user wants more results of the CURRENT list
- reset: ONLY when user explicitly says reset/clear/start over (never use reset for "yes" confirmations)
- paper: user provided an explicit arXiv id (e.g. "paper 2003.10303")
- help: commands/capabilities
- chat: small talk / acknowledgements / ambiguous replies

IMPORTANT DISAMBIGUATION RULES (to avoid wrong behavior):

A) LIST REQUEST vs SINGLE-PAPER REQUEST (critical):
- If user is asking to LIST papers (e.g. "show me papers", "list papers", "find papers", "search papers", "all research papers"),
  then action MUST be "search" (even if words like "show me" appear).
  Examples (=> search):
    - "Show me all research papers written by Andrew Ng"
    - "Show me papers on ai in healthcare"
    - "List cs.AI papers between 2020 and 2022"
    - "Find papers by Andrew Ng between 2020 and 2025"

- If user is asking about ONE specific paper from the current list (by index/title/this/that paper), then action MUST be
  either "summary" or "open" depending on whether they want details vs PDF link.

B) SUMMARY vs OPEN (PDF) for one paper:
- summary = details/abstract/summary/explain/tell me about/data related to
  Examples (=> summary):
    - "show me details of the 1st paper"
    - "tell me about Responsible AI in Healthcare"
    - "summarize the 2nd one"
    - "what is this paper about?"
    - "give me the abstract of the 3rd paper"

- open = PDF/download/link/url/view
  Examples (=> open):
    - "open the 1st paper pdf"
    - "download the 2nd paper"
    - "give me the pdf link for Responsible AI in Healthcare"
    - "view the paper 2103.14954"
    - "open 5 (pdf)"

C) When a list is on screen (session has active search context):
- If user refers to a specific item (index/title/this/that/the paper):
  - If user mentions pdf/download/link/url/view => action="open"
  - Otherwise => action="summary"
- If user is clearly asking for a NEW list (contains plural "papers" + constraints like author/topic/years/categories),
  action="search".

D) arXiv id rule:
- If the user includes an explicit arXiv id (like 2003.10303 or 2103.14954), set action="paper" and fill arxiv_id.
  (Chat service will decide whether to show summary or PDF link using the user's wording.)

E) Confirmation messages:
- If user says just "yes", "okay", "sure", "no", "forget it", "never mind" WITHOUT any search constraints,
  action="chat" and chat_response should be short (or empty string).
- NEVER output action="reset" for "yes its a new search". Use "chat" unless the user explicitly requested reset.

Slot extraction for SEARCH:
- topic: free text topic like "ai in healthcare" (may be empty if only author/categories/years are provided)
- author: if user says "by <name>" or "author <name>" or "written by <name>"
- from_year/to_year: if user says "between <YYYY> and <YYYY>" or "from <YYYY> to <YYYY>" or "<YYYY> only"
- categories: arXiv category codes like "cs.AI", "cs.LG", "stat.ML", "q-bio.QM" (array of strings)

SEARCH example:
Input: "show me research papers uploaded between 2020 and 2022 on ai in healthcare"
Output:
{
  "action": "search",
  "topic": "ai in healthcare",
  "author": "",
  "from_year": 2020,
  "to_year": 2022,
  "categories": [],
  "index": null,
  "title": "",
  "arxiv_id": "",
  "chat_response": ""
}

SINGLE PAPER examples:
Input: "show me details of the 1st paper"
Output: action="summary", index=1

Input: "download the 1st paper"
Output: action="open", index=1

Other commands:
- next: "next", "more results", "next page", "show more results"
- help: "what can you do", "commands", "how does this work", "examples"
- reset: ONLY "reset", "clear", "start over"

Otherwise:
- action="chat" and provide a short friendly chat_response.

Field defaults:
- Use empty string "" for missing strings.
- Use null for missing integers.
- Use [] for missing categories.
"""


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


class LLMClient:
    def __init__(self, endpoint: str, api_key: str, api_version: str, deployment: str) -> None:
        self._enabled = bool(endpoint and api_key and api_version and deployment)
        self.deployment = deployment

        self.client = None
        if self._enabled:
            self.client = AzureOpenAI(
                azure_endpoint=endpoint,
                api_key=api_key,
                api_version=api_version,
            )

    def enabled(self) -> bool:
        return self._enabled and self.client is not None

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.0) -> str:
        if not self.enabled():
            return ""

        resp = self.client.chat.completions.create(
            model=self.deployment,
            messages=messages,
            temperature=temperature,
        )
        return resp.choices[0].message.content or ""

    # ---------------------------------------------------
    # Intent + slot extraction
    # ---------------------------------------------------
    def parse_intent(
        self,
        user_message: str,
        session_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:

        if not self.enabled():
            return {
                "action": "chat",
                "topic": "",
                "author": "",
                "from_year": None,
                "to_year": None,
                "categories": [],
                "index": None,
                "title": "",
                "arxiv_id": "",
                "chat_response": "LLM not configured properly.",
            }

        session_state = session_state or {}

        user_prompt = (
            f"User message:\n{user_message}\n\n"
            f"Session state (for context only):\n{json.dumps(session_state)}"
        )

        raw = self.chat(
            messages=[
                {"role": "system", "content": _INTENT_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
        )

        data = self._safe_parse_json(raw)

        # Normalize and enforce schema
        action = str(data.get("action") or "chat").strip().lower()
        allowed = {"search", "summary", "open", "next", "help", "reset", "paper", "chat"}
        if action not in allowed:
            action = "chat"

        topic = str(data.get("topic") or "").strip()
        author = str(data.get("author") or "").strip()

        fy_raw = data.get("from_year", None)
        ty_raw = data.get("to_year", None)
        from_year = None
        to_year = None
        if fy_raw is not None:
            try:
                from_year = int(fy_raw)
            except Exception:
                from_year = None
        if ty_raw is not None:
            try:
                to_year = int(ty_raw)
            except Exception:
                to_year = None

        categories = data.get("categories") or []
        if not isinstance(categories, list):
            categories = []
        categories = [str(c).strip() for c in categories if str(c).strip()]

        title = str(data.get("title") or "").strip()
        arxiv_id = str(data.get("arxiv_id") or "").strip()
        chat_response = str(data.get("chat_response") or "").strip()

        idx = data.get("index", None)
        index = None
        if idx is not None:
            try:
                index = int(idx)
            except Exception:
                index = None

        # Extra safety guard: never "reset" unless user explicitly asked reset/clear.
        # This prevents accidental "reset" outputs for confirmations.
        msg_low = (user_message or "").strip().lower()
        if action == "reset" and not any(k in msg_low for k in ("reset", "clear", "start over", "restart")):
            action = "chat"
            if not chat_response:
                chat_response = ""

        return {
            "action": action,
            "topic": topic,
            "author": author,
            "from_year": from_year,
            "to_year": to_year,
            "categories": categories,
            "index": index,
            "title": title,
            "arxiv_id": arxiv_id,
            "chat_response": chat_response,
        }

    # ---------------------------------------------------
    # Safe JSON extraction
    # ---------------------------------------------------
    def _safe_parse_json(self, text: str) -> Dict[str, Any]:
        text = (text or "").strip()
        if not text:
            return {}

        try:
            return json.loads(text)
        except Exception:
            pass

        m = _JSON_RE.search(text)
        if not m:
            return {}

        try:
            return json.loads(m.group(0))
        except Exception:
            return {}