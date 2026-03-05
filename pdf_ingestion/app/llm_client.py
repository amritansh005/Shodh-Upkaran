"""
llm_client.py — Minimal Azure OpenAI client for the pdf_ingestion service.
Used only by QAService to generate answers from retrieved context.
"""

from __future__ import annotations

import logging
from typing import Dict, List

from openai import AzureOpenAI

logger = logging.getLogger(__name__)


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
