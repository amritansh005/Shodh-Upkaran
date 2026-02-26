from typing import List, Optional, Literal
from pydantic import BaseModel, Field


SortBy = Literal["relevance", "lastUpdatedDate", "submittedDate"]
SortOrder = Literal["ascending", "descending"]


class SearchRequest(BaseModel):
    # Free-text topic (optional). If provided, we search in title+abstract (ti/abs).
    # If omitted, you can still search by author/date/category alone.
    topic: Optional[str] = Field(
        default=None,
        description="Search topic, e.g., 'ai for healthcare' (optional if other filters are set)",
    )

    # Optional structured filters
    author: Optional[str] = Field(
        default=None,
        description='Author name filter, e.g., "Andrew Ng" (maps to arXiv au:"...")',
    )

    # Year-only ranges (inclusive). We convert to submittedDate range.
    from_year: Optional[int] = Field(default=None, ge=1900, le=2100, description="Start year (inclusive)")
    to_year: Optional[int] = Field(default=None, ge=1900, le=2100, description="End year (inclusive)")

    max_results: int = Field(10, ge=1, le=500, description="Number of results to return (1-500)")
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
