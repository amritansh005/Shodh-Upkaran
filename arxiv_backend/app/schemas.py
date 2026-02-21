from typing import List, Optional, Literal
from pydantic import BaseModel, Field


SortBy = Literal["relevance", "lastUpdatedDate", "submittedDate"]
SortOrder = Literal["ascending", "descending"]


class SearchRequest(BaseModel):
    topic: str = Field(..., min_length=2, description="Search topic, e.g., 'ai for healthcare'")
    max_results: int = Field(10, ge=1, le=50, description="Number of results to return (1-50)")
    start: int = Field(0, ge=0, description="Pagination offset")
    sort_by: SortBy = Field("relevance", description="arXiv sort option")
    sort_order: SortOrder = Field("descending", description="Sort order")

    # Optional category filters like: cs.AI, cs.LG, cs.CL, stat.ML, q-bio.QM
    categories: Optional[List[str]] = Field(default=None, description="arXiv category codes")


class Paper(BaseModel):
    paper_id: str  # "arxiv:xxxx.xxxxx"
    arxiv_id: str  # "xxxx.xxxxx" (or old style)
    title: str
    authors: List[str]
    abstract: str
    categories: List[str]
    published_date: str
    updated_date: str
    pdf_url: str
    abs_url: str
    source: Literal["arXiv"] = "arXiv"


class SearchResponse(BaseModel):
    query: str
    start: int
    max_results: int
    total_results: Optional[int] = None
    results: List[Paper]


class ErrorResponse(BaseModel):
    detail: str
