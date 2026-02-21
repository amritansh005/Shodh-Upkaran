from __future__ import annotations

import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from fastapi import FastAPI
import uvicorn

from chatbot_service.schemas import ChatRequest, ChatResponse
from chatbot_service.services.arxiv_backend_client import ArxivBackendClient
from chatbot_service.services.llm_client import LLMClient
from chatbot_service.services.chat_service import ChatService
from chatbot_service.services.session_store import InMemorySessionStore


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file="chatbot_service/.env", env_file_encoding="utf-8", extra="ignore")

    chatbot_host: str = "127.0.0.1"
    chatbot_port: int = 9000

    arxiv_backend_base_url: str = "http://127.0.0.1:8000"

    azure_openai_endpoint: str = ""
    azure_openai_api_key: str = ""
    azure_openai_deployment: str = "gpt-4o"
    azure_openai_api_version: str = "2025-01-01-preview"


settings = Settings()

app = FastAPI(title="Chatbot Service (Option 1)", version="0.1")

# Dependencies
arxiv_client = ArxivBackendClient(base_url=settings.arxiv_backend_base_url)
llm_client = LLMClient(
    endpoint=settings.azure_openai_endpoint,
    api_key=settings.azure_openai_api_key,
    api_version=settings.azure_openai_api_version,
    deployment=settings.azure_openai_deployment,
)
session_store = InMemorySessionStore()
chat_service = ChatService(arxiv=arxiv_client, llm=llm_client, store=session_store)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "llm_enabled": llm_client.enabled(),
        "arxiv_backend_base_url": settings.arxiv_backend_base_url,
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    reply, results, meta = await chat_service.handle(req.session_id, req.message)
    return ChatResponse(session_id=req.session_id, reply=reply, results=results, meta=meta)


if __name__ == "__main__":
    uvicorn.run(app, host=settings.chatbot_host, port=settings.chatbot_port)
