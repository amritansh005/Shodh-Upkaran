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
  "action": "search" | "open" | "next" | "help" | "reset" | "paper" | "chat",
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


Rules:

IMPORTANT CONTEXT RULE (when a list is on screen):
- If the session state indicates there is an active search/list (e.g., last_results is non-empty),
  and the user uses OPEN/READ language such as "open", "read", "summarize", "details", "tell me about",
  then set action="open" and extract selectors (index/title/author/from_year/to_year/topic) from the message.
  Do NOT set action="search" for such messages.

- If user asks to search, extract search constraints:

  - topic (free-text topic like "ai in healthcare")
  - author if user says "by <name>" or "author <name>"
  - year range if user says "between <YYYY> and <YYYY>" or "from <YYYY> to <YYYY>"
  - categories if user mentions arXiv category codes like "cs.AI", "cs.LG", "stat.ML", "q-bio.QM"
  Example:
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

- If user message is just a topic like "ai in healthcare",
  treat it as action="search" with topic="ai in healthcare".

- If user says only an author constraint like "papers by Andrew Ng",
  action="search", topic="" and author="Andrew Ng".

- If user says "next", "more results", "next page", action="next".

- If user says "open 5" or "show me the 2nd paper",
  action="open" and index=number (1-based).

- If user says "open <TITLE>" or "show me the paper titled <TITLE>",
  action="open" and title=<TITLE> (and index=null).

- If user says "paper 2103.14954",
  action="paper" and arxiv_id="2103.14954".

- If user asks what you can do or commands, action="help".

- If user says reset or clear, action="reset".

- Otherwise action="chat" and provide short friendly chat_response.

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
        if action not in {"search", "open", "next", "help", "reset", "paper", "chat"}:
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