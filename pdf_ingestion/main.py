"""
main.py — pdf_ingestion service entry point.

Run with:
    cd Shodh Upkaran
    python -m pdf_ingestion.main
or:
    uvicorn pdf_ingestion.main:app --port 7000
"""

from __future__ import annotations

import os
import shutil
import threading
import logging

# ---------------------------------------------------------------------------
# Windows symlink fix — MUST appear before huggingface_hub is imported.
#
# HuggingFace Hub's cache system creates symlinks to avoid duplicating large
# model files. On Windows this requires Developer Mode or Administrator rights.
# Without them every model download crashes with:
#   OSError: [WinError 1314] A required privilege is not held by the client
#
# Fix: monkey-patch huggingface_hub._create_symlink so that when a symlink
# would fail on Windows we fall back to a plain file copy instead.
# This is safe — it just uses more disk space (no deduplication), which is
# the documented "degraded mode" HF Hub mentions in its own warning.
# ---------------------------------------------------------------------------
import huggingface_hub.file_download as _hf_fd


def _create_symlink_or_copy(src: str, dst: str, new_blob: bool = False) -> None:
    """Drop-in replacement for huggingface_hub._create_symlink.
    Tries a real symlink first; on WinError 1314 (privilege not held) it
    falls back to copying the file so the cache still works without
    Developer Mode or Administrator rights.
    """
    try:
        os.symlink(src, dst)
    except OSError as exc:
        if getattr(exc, "winerror", None) == 1314:
            # Resolve src relative to dst's directory (HF uses relative symlinks)
            src_abs = os.path.normpath(os.path.join(os.path.dirname(dst), src))
            if os.path.isfile(src_abs) and not os.path.exists(dst):
                shutil.copy2(src_abs, dst)
        else:
            raise


_hf_fd._create_symlink = _create_symlink_or_copy

import uvicorn
from fastapi import FastAPI
from pydantic_settings import BaseSettings, SettingsConfigDict

from pdf_ingestion.app.paper_store import PaperStore
from pdf_ingestion.app.qa_service import QAService
from pdf_ingestion.app.llm_client import LLMClient
from pdf_ingestion.app.embedder import get_embedder
from pdf_ingestion.app.routes import router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file="pdf_ingestion/.env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    pdf_ingestion_host: str = "127.0.0.1"
    pdf_ingestion_port: int = 7000

    postgres_dsn: str = ""

    azure_openai_endpoint: str = ""
    azure_openai_api_key: str = ""
    azure_openai_deployment: str = "gpt-4o"
    azure_openai_api_version: str = "2025-01-01-preview"

    # Safety flag:
    # keep Marker preload disabled until thread-safe singleton init is confirmed
    preload_marker_on_startup: bool = False


settings = Settings()

app = FastAPI(title="PDF Ingestion Service", version="1.0")
app.include_router(router)


def _preload_embedder() -> None:
    """
    Eagerly load only the embedding model in a background thread so it is warm
    before the first paper is opened.

    Kept separate from Marker preload to avoid concurrent Marker initialization
    races during startup + first request.
    """
    try:
        logger.info("[pdf_ingestion] Pre-loading embedding model...")
        get_embedder()._load()
        logger.info("[pdf_ingestion] Embedding model ready.")
    except Exception as e:
        logger.warning("[pdf_ingestion] Embedder pre-load failed (non-fatal): %s", e)


def _preload_marker() -> None:
    """
    Optionally preload Marker after startup.

    This should only be enabled once vision_extractor._get_converter() has been
    made thread-safe with a module-level lock.
    """
    try:
        logger.info("[pdf_ingestion] Pre-loading Marker models onto GPU...")
        from pdf_ingestion.app.vision_extractor import _get_converter

        _get_converter()
        logger.info("[pdf_ingestion] Marker models ready on GPU.")
    except Exception as e:
        logger.warning("[pdf_ingestion] Marker pre-load failed (non-fatal): %s", e)


@app.on_event("startup")
def startup():
    if not settings.postgres_dsn:
        logger.error(
            "POSTGRES_DSN is not set in pdf_ingestion/.env — "
            "the service will start but all endpoints will fail."
        )
        app.state.paper_store = None
        app.state.qa_service = None
        return

    app.state.settings = settings
    paper_store = PaperStore(dsn=settings.postgres_dsn)
    paper_store.init_db()
    app.state.paper_store = paper_store

    llm = LLMClient(
        endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        api_version=settings.azure_openai_api_version,
        deployment=settings.azure_openai_deployment,
    )
    app.state.qa_service = QAService(store=paper_store, llm_client=llm)

    logger.info(
        "[pdf_ingestion] Started on port %d | DB: connected | LLM: %s",
        settings.pdf_ingestion_port,
        "enabled" if llm.enabled() else "disabled",
    )

    # Warm the embedder in background.
    # This is safe and avoids delaying server readiness.
    threading.Thread(target=_preload_embedder, daemon=True).start()

    # Marker preload is optional.
    # Leave disabled for now to avoid startup-time races with first ingest request.
    if settings.preload_marker_on_startup:
        threading.Thread(target=_preload_marker, daemon=True).start()
    else:
        logger.info(
            "[pdf_ingestion] Marker startup preload disabled; will load on first use."
        )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "db_connected": app.state.paper_store is not None,
        "llm_enabled": (
            app.state.qa_service._llm.enabled()
            if app.state.qa_service else False
        ),
        "embedder_ready": get_embedder()._model is not None,
    }


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=settings.pdf_ingestion_host,
        port=settings.pdf_ingestion_port,
    )