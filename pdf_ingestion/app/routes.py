"""
routes.py — FastAPI routes for the pdf_ingestion service.

Endpoints
---------
POST /ingest
    Body: paper metadata dict
    Triggers full pipeline: download → extract → chunk → embed → store.
    Returns: { status, message, arxiv_id, chunks, used_ocr }

GET  /status/{arxiv_id}
    Returns current processing status for a paper.
    { arxiv_id, status }   status: pending | processing | ready | failed | not_found

POST /ask
    Body: { arxiv_id, question, paper_title }
    Returns: { arxiv_id, answer }
"""

from __future__ import annotations

from typing import List

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from pdf_ingestion.app.ingest_pipeline import ingest_paper

router = APIRouter()


# ------------------------------------------------------------------
# Request / Response schemas
# ------------------------------------------------------------------

class IngestRequest(BaseModel):
    arxiv_id: str
    title: str = ""
    authors: List[str] = Field(default_factory=list)
    pdf_url: str = ""
    abstract: str = ""
    published: str = ""


class IngestResponse(BaseModel):
    status: str  # "ready" | "already_ready" | "failed"
    message: str  # shown directly to user
    arxiv_id: str
    chunks: int = 0
    used_ocr: bool = False


class StatusResponse(BaseModel):
    arxiv_id: str
    status: str  # "pending" | "processing" | "ready" | "failed" | "not_found"


class AskRequest(BaseModel):
    arxiv_id: str
    question: str
    paper_title: str = ""


class AskResponse(BaseModel):
    arxiv_id: str
    answer: str


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _require_store(request: Request):
    store = getattr(request.app.state, "paper_store", None)
    if store is None:
        raise HTTPException(
            status_code=503,
            detail="Paper store not available (POSTGRES_DSN missing or DB init failed).",
        )
    return store


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------

@router.post("/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest, request: Request):
    store = _require_store(request)

    paper_dict = {
        "arxiv_id": req.arxiv_id,
        "title": req.title,
        "authors": req.authors,
        "pdf_url": req.pdf_url,
        "abstract": req.abstract,
        "published": req.published,
    }

    result = await ingest_paper(paper=paper_dict, store=store)
    return IngestResponse(**result)


@router.get("/status/{arxiv_id}", response_model=StatusResponse)
def status(arxiv_id: str, request: Request):
    store = _require_store(request)

    st = store.get_paper_status(arxiv_id)
    return StatusResponse(
        arxiv_id=arxiv_id,
        status=st if st else "not_found",
    )


@router.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, request: Request):
    qa_svc = getattr(request.app.state, "qa_service", None)
    if qa_svc is None:
        raise HTTPException(status_code=503, detail="QA service not available.")

    answer = qa_svc.answer(
        arxiv_id=req.arxiv_id,
        question=req.question,
        paper_title=req.paper_title,
    )
    return AskResponse(arxiv_id=req.arxiv_id, answer=answer)