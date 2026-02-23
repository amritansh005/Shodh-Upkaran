from __future__ import annotations

import json
import re
from typing import List, Dict, Any, Optional

from openai import AzureOpenAI


_INTENT_SYSTEM_PROMPT = """
You are an intent and slot extractor for an arXiv research assistant chatbot.

Return ONLY valid JSON. No markdown. No explanation. No extra text.

Schema:

{
  "action": "search" | "open" | "next" | "help" | "reset" | "paper" | "chat",
  "topic": string,
  "index": integer or null,
  "title": string,
  "arxiv_id": string,
  "chat_response": string
}

Rules:

- If user asks to search, extract ONLY the topic.
  Example:
    Input: "can you search for research papers on ai in healthcare"
    Output:
    {
      "action": "search",
      "topic": "ai in healthcare",
      "index": null,
      "title": "",
      "arxiv_id": "",
      "chat_response": ""
    }

- If user message is just a topic like "ai in healthcare",
  treat it as action="search" with topic="ai in healthcare".

- If user says "next", "more results", "next page", action="next".

- If user says "open 5" or "show me the 2nd paper",
  action="open" and index=number (1-based).

- If user says "open <TITLE>" or "show me the paper titled <TITLE>",
  action="open" and title=<TITLE> (and index=null).
  Examples:
    Input: "open Responsible AI in Healthcare"
    Output:
    {
      "action": "open",
      "topic": "",
      "index": null,
      "title": "Responsible AI in Healthcare",
      "arxiv_id": "",
      "chat_response": ""
    }

- If user says "paper 2103.14954",
  action="paper" and arxiv_id="2103.14954".

- If user asks what you can do or commands, action="help".

- If user says reset or clear, action="reset".

- Otherwise action="chat" and provide short friendly chat_response.

ALWAYS include all fields.
Use empty string "" for missing strings.
Use null for missing index.
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