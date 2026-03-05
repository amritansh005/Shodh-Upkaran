"""
pdf_ingestion_client.py — HTTP client for the pdf_ingestion service.

chatbot_service uses this to:
  - POST /ingest  — trigger download + processing for a paper
  - GET  /status/{arxiv_id} — check if already ingested
  - POST /ask     — ask a question about an ingested paper

This is the ONLY pdf-related file in chatbot_service.
All actual PDF, embedding, and DB logic lives in the pdf_ingestion service.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import httpx

logger = logging.getLogger(__name__)


class PDFIngestionClient:
    def __init__(self, base_url: str = "http://127.0.0.1:7000", timeout: float = 600.0) -> None:
        # timeout is long because ingest can take time (download + OCR + embedding)
        self._base_url = base_url.rstrip("/")
        self._timeout  = timeout

    # ------------------------------------------------------------------
    # /status/{arxiv_id}
    # ------------------------------------------------------------------

    async def get_status(self, arxiv_id: str) -> str:
        """
        Returns the ingestion status for a paper:
        'ready' | 'processing' | 'pending' | 'failed' | 'not_found' | 'unreachable'
        """
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{self._base_url}/status/{arxiv_id}")
                resp.raise_for_status()
                return resp.json().get("status", "not_found")
        except httpx.ConnectError:
            logger.error("[PDF_CLIENT] Cannot connect to pdf_ingestion service at %s", self._base_url)
            return "unreachable"
        except Exception as e:
            logger.error("[PDF_CLIENT] get_status failed: %s", e)
            return "unreachable"

    # ------------------------------------------------------------------
    # POST /ingest
    # ------------------------------------------------------------------

    async def ingest(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        """
        Trigger ingestion for a paper. Returns the result dict:
        {
            "status":   "ready" | "already_ready" | "failed",
            "message":  str,    <- shown directly to user
            "arxiv_id": str,
            "chunks":   int,
            "used_ocr": bool,
        }

        If the pdf_ingestion service is unreachable, returns a 'failed' result
        with a specific error message (never falls back to showing the PDF link).
        """
        authors = paper.get("authors") or []
        if isinstance(authors, list):
            authors_list = [str(a) for a in authors]
        else:
            authors_list = [str(authors)] if authors else []

        payload = {
            "arxiv_id":  str(paper.get("arxiv_id") or paper.get("id") or ""),
            "title":     str(paper.get("title") or ""),
            "authors":   authors_list,
            "pdf_url":   self._resolve_pdf_url(paper),
            "abstract":  str(paper.get("abstract") or paper.get("summary") or ""),
            "published": str(paper.get("published") or paper.get("published_date") or ""),
        }

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(f"{self._base_url}/ingest", json=payload)
                resp.raise_for_status()
                return resp.json()

        except httpx.ConnectError:
            logger.error("[PDF_CLIENT] Cannot connect to pdf_ingestion service at %s", self._base_url)
            return {
                "status":   "failed",
                "message":  (
                    "The PDF ingestion server is not reachable. "
                    "Please make sure the pdf_ingestion service is running on "
                    f"{self._base_url} and try again."
                ),
                "arxiv_id": payload["arxiv_id"],
                "chunks":   0,
                "used_ocr": False,
            }

        except httpx.TimeoutException:
            return {
                "status":   "failed",
                "message":  (
                    "The ingestion request timed out. The PDF may be very large or "
                    "the server is under heavy load. Please try again."
                ),
                "arxiv_id": payload["arxiv_id"],
                "chunks":   0,
                "used_ocr": False,
            }

        except Exception as e:
            logger.error("[PDF_CLIENT] ingest call failed: %s", e)
            return {
                "status":   "failed",
                "message":  f"An unexpected error occurred while contacting the ingestion service: {e}",
                "arxiv_id": payload["arxiv_id"],
                "chunks":   0,
                "used_ocr": False,
            }

    # ------------------------------------------------------------------
    # POST /ask
    # ------------------------------------------------------------------

    async def ask(
        self,
        arxiv_id: str,
        question: str,
        paper_title: str = "",
    ) -> str:
        """
        Ask a question about an ingested paper.
        Returns the answer string.
        """
        payload = {
            "arxiv_id":    arxiv_id,
            "question":    question,
            "paper_title": paper_title,
        }
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(f"{self._base_url}/ask", json=payload)
                resp.raise_for_status()
                return resp.json().get("answer", "No answer returned.")

        except httpx.ConnectError:
            return (
                "The PDF ingestion server is not reachable. "
                "Please make sure the pdf_ingestion service is running."
            )
        except httpx.TimeoutException:
            return "The Q&A request timed out. Please try again."
        except Exception as e:
            logger.error("[PDF_CLIENT] ask failed: %s", e)
            return f"An unexpected error occurred while getting the answer: {e}"

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_pdf_url(paper: Dict[str, Any]) -> str:
        pdf = (
            paper.get("pdf_url")
            or paper.get("pdf")
            or paper.get("pdfLink")
            or paper.get("pdf_link")
            or ""
        )
        if isinstance(pdf, dict):
            pdf = pdf.get("href") or pdf.get("url") or ""
        pdf = str(pdf or "").strip()
        if pdf:
            return pdf
        arxiv_id = str(paper.get("arxiv_id") or paper.get("id") or "").strip()
        if arxiv_id:
            return f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        return ""