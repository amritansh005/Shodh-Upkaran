from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# Mirror the fields your arxiv_backend returns (Paper)
class Paper(BaseModel):
    arxiv_id: str
    title: str
    authors: List[str] = Field(default_factory=list)
    abstract: str = ""
    published: Optional[str] = None
    updated: Optional[str] = None
    pdf_url: Optional[str] = None
    categories: List[str] = Field(default_factory=list)


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=3)
    message: str = Field(..., min_length=1)


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    results: Optional[List[Paper]] = None
    meta: Dict[str, Any] = Field(default_factory=dict)
