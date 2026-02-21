from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.schemas import Paper
from app.services.arxiv_client import ArxivClient


def _extract_arxiv_id(entry: Dict[str, Any]) -> str:
    # entry.id is like: http://arxiv.org/abs/2308.01234v1
    entry_id = entry.get("id", "")
    # take last part after /abs/
    if "/abs/" in entry_id:
        tail = entry_id.split("/abs/")[-1]
        # remove version suffix vN
        return tail.split("v")[0]
    return entry_id


def _extract_pdf_url(entry: Dict[str, Any]) -> str:
    # arXiv entries include links; one is usually type=application/pdf
    links = entry.get("links", []) or []
    for l in links:
        if (l.get("type") == "application/pdf") or ("pdf" in (l.get("href", "") or "").lower()):
            href = l.get("href")
            if href:
                return href
    # fallback: convert abs -> pdf
    abs_url = entry.get("link", "")
    if "/abs/" in abs_url:
        return abs_url.replace("/abs/", "/pdf/") + ".pdf"
    return abs_url


def _extract_categories(entry: Dict[str, Any]) -> List[str]:
    tags = entry.get("tags", []) or []
    cats = []
    for t in tags:
        term = t.get("term")
        if term:
            cats.append(term)
    # Keep unique order
    seen = set()
    out = []
    for c in cats:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def normalize_entry(entry: Dict[str, Any]) -> Paper:
    arxiv_id = _extract_arxiv_id(entry)

    title = (entry.get("title") or "").replace("\n", " ").strip()
    abstract = (entry.get("summary") or "").replace("\n", " ").strip()

    authors_raw = entry.get("authors", []) or []
    authors = []
    for a in authors_raw:
        name = a.get("name")
        if name:
            authors.append(name)

    abs_url = entry.get("link", "") or entry.get("id", "")
    pdf_url = _extract_pdf_url(entry)

    published = (entry.get("published") or "").strip()
    updated = (entry.get("updated") or "").strip()

    categories = _extract_categories(entry)

    return Paper(
        paper_id=f"arxiv:{arxiv_id}",
        arxiv_id=arxiv_id,
        title=title,
        authors=authors,
        abstract=abstract,
        categories=categories,
        published_date=published,
        updated_date=updated,
        pdf_url=pdf_url,
        abs_url=abs_url,
        source="arXiv",
    )


class SearchService:
    def __init__(self, client: ArxivClient) -> None:
        self.client = client

    async def search(
        self,
        topic: str,
        start: int,
        max_results: int,
        sort_by: str,
        sort_order: str,
        categories: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        total, entries = await self.client.search(
            topic=topic,
            start=start,
            max_results=max_results,
            sort_by=sort_by,
            sort_order=sort_order,
            categories=categories,
        )
        papers = [normalize_entry(e) for e in entries]
        return {"total": total, "papers": papers}

    async def get_paper(self, arxiv_id: str) -> Optional[Paper]:
        entry = await self.client.get_by_id(arxiv_id)
        if not entry:
            return None
        return normalize_entry(entry)