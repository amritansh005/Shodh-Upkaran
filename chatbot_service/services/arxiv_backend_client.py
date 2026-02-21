from __future__ import annotations

from typing import Any, Dict, Optional, List

import httpx

from chatbot_service.schemas import Paper


class ArxivBackendClient:
    def __init__(self, base_url: str, timeout_s: float = 300.0) -> None:
        self.base_url = base_url.rstrip("/")

        # Structured timeout (important!)
        self.timeout = httpx.Timeout(
            connect=5.0,   # time to connect to backend
            read=300.0,    # time waiting for backend response (must exceed retry window)
            write=10.0,
            pool=5.0,
        )

    async def search(
        self,
        topic: str,
        start: int = 0,
        max_results: int = 10,
        sort_by: str = "relevance",
        sort_order: str = "descending",
        categories: Optional[List[str]] = None,
    ) -> Dict[str, Any]:

        payload = {
            "topic": topic,
            "start": start,
            "max_results": max_results,
            "sort_by": sort_by,
            "sort_order": sort_order,
            "categories": categories,
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                r = await client.post(f"{self.base_url}/search", json=payload)

        except httpx.ConnectError:
            return {
                "papers": [],
                "total_results": 0,
                "error": "backend_unreachable",
            }

        except httpx.ReadTimeout:
            return {
                "papers": [],
                "total_results": 0,
                "error": "backend_timeout",
            }

        except httpx.RequestError:
            return {
                "papers": [],
                "total_results": 0,
                "error": "network_error",
            }

        # Handle backend error responses explicitly
        if r.status_code == 502:
            return {
                "papers": [],
                "total_results": 0,
                "error": "arxiv_unavailable",
            }

        if r.status_code == 503:
            return {
                "papers": [],
                "total_results": 0,
                "error": "arxiv_overloaded",
            }

        if r.status_code >= 400:
            return {
                "papers": [],
                "total_results": 0,
                "error": f"backend_error_{r.status_code}",
            }

        data = r.json()

        raw_results = data.get("results") or data.get("papers") or []

        papers: List[Paper] = []
        for p in raw_results:
            papers.append(
                Paper(
                    arxiv_id=p.get("arxiv_id"),
                    title=p.get("title", "") or "",
                    authors=p.get("authors") or [],
                    abstract=p.get("abstract") or "",
                    published=p.get("published_date") or p.get("published"),
                    updated=p.get("updated_date") or p.get("updated"),
                    pdf_url=p.get("pdf_url"),
                    categories=p.get("categories") or [],
                )
            )

        return {
            "papers": papers,
            "total_results": data.get("total_results") or data.get("total") or len(papers),
        }

    async def get_paper(self, arxiv_id: str) -> Optional[Paper]:
        arxiv_id = arxiv_id.strip()

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                r = await client.get(f"{self.base_url}/paper/{arxiv_id}")

        except httpx.RequestError:
            return None

        if r.status_code == 404:
            return None

        if r.status_code >= 400:
            return None

        data = r.json()

        return Paper(
            arxiv_id=data.get("arxiv_id"),
            title=data.get("title", "") or "",
            authors=data.get("authors") or [],
            abstract=data.get("abstract") or "",
            published=data.get("published_date") or data.get("published"),
            updated=data.get("updated_date") or data.get("updated"),
            pdf_url=data.get("pdf_url"),
            categories=data.get("categories") or [],
        )