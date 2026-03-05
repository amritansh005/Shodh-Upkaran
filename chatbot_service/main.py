from __future__ import annotations

import logging

from pydantic_settings import BaseSettings, SettingsConfigDict
from fastapi import FastAPI
import uvicorn

from chatbot_service.schemas import ChatRequest, ChatResponse
from chatbot_service.services.arxiv_backend_client import ArxivBackendClient
from chatbot_service.services.llm_client import LLMClient
from chatbot_service.services.chat_service import ChatService
from chatbot_service.services.session_store import InMemorySessionStore
from chatbot_service.services.pdf_ingestion_client import PDFIngestionClient

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file="chatbot_service/.env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    chatbot_host: str = "127.0.0.1"
    chatbot_port: int = 9000

    arxiv_backend_base_url: str = "http://127.0.0.1:8000"

    azure_openai_endpoint: str = ""
    azure_openai_api_key: str = ""
    azure_openai_deployment: str = "gpt-4o"
    azure_openai_api_version: str = "2025-01-01-preview"

    # Pagination / progressive prefetch
    page_size_default: int = 10
    prefetch_max_results: int = 200
    prefetch_chunk_size: int = 200
    hard_total_cap: int = 5000

    # PDF ingestion service URL
    pdf_ingestion_base_url: str = "http://127.0.0.1:7000"


settings = Settings()

app = FastAPI(title="Chatbot Service", version="0.1")

# Core dependencies
arxiv_client = ArxivBackendClient(base_url=settings.arxiv_backend_base_url)
llm_client = LLMClient(
    endpoint=settings.azure_openai_endpoint,
    api_key=settings.azure_openai_api_key,
    api_version=settings.azure_openai_api_version,
    deployment=settings.azure_openai_deployment,
)
session_store = InMemorySessionStore()

# PDF ingestion client — thin HTTP wrapper, no PDF code here
pdf_ingestion_client = PDFIngestionClient(
    base_url=settings.pdf_ingestion_base_url
) if settings.pdf_ingestion_base_url else None

chat_service = ChatService(
    arxiv=arxiv_client,
    llm=llm_client,
    store=session_store,
    page_size_default=settings.page_size_default,
    prefetch_max_results=settings.prefetch_max_results,
    prefetch_chunk_size=settings.prefetch_chunk_size,
    hard_total_cap=settings.hard_total_cap,
    pdf_ingestion_client=pdf_ingestion_client,
)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "llm_enabled": llm_client.enabled(),
        "arxiv_backend_base_url": settings.arxiv_backend_base_url,
        "pdf_ingestion_base_url": settings.pdf_ingestion_base_url,
        "page_size_default": settings.page_size_default,
        "prefetch_max_results": settings.prefetch_max_results,
        "prefetch_chunk_size": settings.prefetch_chunk_size,
        "hard_total_cap": settings.hard_total_cap,
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    reply, results, meta = await chat_service.handle_message(req.session_id, req.message)
    return ChatResponse(session_id=req.session_id, reply=reply, results=results, meta=meta)


if __name__ == "__main__":
    uvicorn.run(app, host=settings.chatbot_host, port=settings.chatbot_port)