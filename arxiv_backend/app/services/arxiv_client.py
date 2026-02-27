# app/services/arxiv_client.py

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import asyncio
import logging

import httpx
import feedparser

from app.config import settings
from app.utils.cache import AsyncTokenBucketRateLimiter, AsyncConcurrencyLimiter

logger = logging.getLogger(__name__)


class ArxivClient:
    """
    arXiv API client using the official Atom feed endpoint:
    http://export.arxiv.org/api/query

    Outgoing throttles (GLOBAL):
      - concurrency limit (semaphore)
      - rate limit (token bucket)
    """
    def __init__(self) -> None:
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                timeout=settings.http_timeout_seconds,
                connect=10.0,
                read=settings.http_timeout_seconds,
                write=10.0,
                pool=10.0,
            ),
            headers={"User-Agent": settings.user_agent},
            follow_redirects=True,
        )

        # Global throttles for outgoing traffic
        self._concurrency = AsyncConcurrencyLimiter(settings.outgoing_max_concurrency)
        self._rate = AsyncTokenBucketRateLimiter(
            rate=settings.outgoing_rps,
            capacity=settings.outgoing_burst,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    def build_search_query(
        self,
        topic: str = "",
        author: Optional[str] = None,
        from_year: Optional[int] = None,
        to_year: Optional[int] = None,
        categories: Optional[List[str]] = None,
    ) -> str:
        """Build an arXiv API `search_query` string.

        We keep your original behavior (title+abstract) but add optional:
          - author (au:"...")
          - category filters (cat:...)
          - submittedDate range (YYYYMMDDhhmm)

        All provided constraints are AND'ed together.
        """
        parts: List[str] = []

        safe_topic = (topic or "").strip().replace('"', "").replace("'", "")
        if safe_topic:
            parts.append(f'(ti:"{safe_topic}" OR abs:"{safe_topic}")')

        safe_author = (author or "").strip().replace('"', "").replace("'", "")
        if safe_author:
            parts.append(f'au:"{safe_author}"')

        # submittedDate range is inclusive; we map years to full-year windows.
        if from_year is not None or to_year is not None:
            fy = int(from_year) if from_year is not None else 1900
            ty = int(to_year) if to_year is not None else 2100
            if ty < fy:
                fy, ty = ty, fy
            start = f"{fy:04d}01010000"
            end = f"{ty:04d}12312359"
            parts.append(f"submittedDate:[{start} TO {end}]")

        if categories:
            cat_parts = [f"cat:{c.strip()}" for c in categories if (c or "").strip()]
            if cat_parts:
                cat_q = " OR ".join(cat_parts)
                parts.append(f"({cat_q})")

        # If nothing is provided, return empty (caller should validate)
        if not parts:
            return ""

        if len(parts) == 1:
            return parts[0]

        return " AND ".join(f"({p})" for p in parts)

    async def _get_throttled(self, url: str, params: Dict[str, Any]) -> httpx.Response:
        """
        Applies:
          1) concurrency limit
          2) rate limit
          3) retries on 429 + transient 5xx/timeouts
          4) overall deadline across retry chain (attempts + sleeps)
        """
        max_retries = settings.outgoing_max_retries
        base = settings.outgoing_retry_backoff_base_seconds

        # NEW: overall budget for the whole retry chain
        deadline_s = float(getattr(settings, "outgoing_total_deadline_seconds", 0) or 0)
        loop = asyncio.get_running_loop()
        start_t = loop.time()

        def remaining_budget() -> float:
            if deadline_s <= 0:
                return 1e9  # effectively no deadline
            return deadline_s - (loop.time() - start_t)

        attempt = 0
        while True:
            attempt += 1

            rem = remaining_budget()
            if rem <= 0:
                raise httpx.ReadTimeout(f"overall upstream deadline exceeded ({deadline_s:.1f}s)")

            try:
                logger.info("THROTTLE: waiting for slot + token")
                async with self._concurrency:
                    await self._rate.acquire()
                    logger.info("THROTTLE: acquired slot + token → sending upstream request")

                    # Bound this request by remaining budget (and also your standard timeout)
                    req_timeout = min(rem, float(settings.http_timeout_seconds))
                    resp = await self._client.get(url, params=params, timeout=req_timeout)

                # If upstream rate-limits you, honor Retry-After if present
                if resp.status_code == 429:
                    retry_after = resp.headers.get("Retry-After")
                    if attempt <= max_retries:
                        if retry_after and retry_after.isdigit():
                            sleep_s = float(retry_after)
                        else:
                            sleep_s = base * (2 ** (attempt - 1))

                        # Don't sleep past overall budget
                        rem = remaining_budget()
                        if rem <= 0:
                            raise httpx.ReadTimeout(f"overall upstream deadline exceeded ({deadline_s:.1f}s)")
                        sleep_s = min(sleep_s, max(0.0, rem))

                        logger.warning(
                            "UPSTREAM 429. retrying in %.2fs attempt=%d/%d",
                            sleep_s,
                            attempt,
                            max_retries,
                        )
                        await asyncio.sleep(sleep_s)
                        continue

                # transient errors
                if resp.status_code in (502, 503, 504):
                    if attempt <= max_retries:
                        sleep_s = base * (2 ** (attempt - 1))

                        rem = remaining_budget()
                        if rem <= 0:
                            raise httpx.ReadTimeout(f"overall upstream deadline exceeded ({deadline_s:.1f}s)")
                        sleep_s = min(sleep_s, max(0.0, rem))

                        logger.warning(
                            "UPSTREAM %d. retrying in %.2fs attempt=%d/%d",
                            resp.status_code,
                            sleep_s,
                            attempt,
                            max_retries,
                        )
                        await asyncio.sleep(sleep_s)
                        continue

                resp.raise_for_status()
                return resp

            except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.RemoteProtocolError, httpx.ConnectError) as e:
                if attempt <= max_retries:
                    sleep_s = base * (2 ** (attempt - 1))

                    rem = remaining_budget()
                    if rem <= 0:
                        raise httpx.ReadTimeout(f"overall upstream deadline exceeded ({deadline_s:.1f}s)") from e
                    sleep_s = min(sleep_s, max(0.0, rem))

                    logger.warning(
                        "UPSTREAM network error (%s). retrying in %.2fs attempt=%d/%d",
                        str(e),
                        sleep_s,
                        attempt,
                        max_retries,
                    )
                    await asyncio.sleep(sleep_s)
                    continue
                raise

    async def search(
        self,
        topic: str,
        author: Optional[str],
        from_year: Optional[int],
        to_year: Optional[int],
        start: int,
        max_results: int,
        sort_by: str,
        sort_order: str,
        categories: Optional[List[str]] = None,
    ) -> Tuple[Optional[int], List[Dict[str, Any]], str]:
        search_query = self.build_search_query(
            topic=topic,
            author=author,
            from_year=from_year,
            to_year=to_year,
            categories=categories,
        )

        if not search_query:
            # Caller should validate, but keep a defensive fallback.
            return 0, [], ""

        params = {
            "search_query": search_query,
            "start": start,
            "max_results": max_results,
            "sortBy": sort_by,
            "sortOrder": sort_order,
        }

        # Helps confirm coalescing (you should see this once per coalesced key)
        logger.info("UPSTREAM arXiv search called. params=%s", params)

        # DEV ONLY delay (your existing coalescing test hook)
        delay = getattr(settings, "coalesce_test_delay_seconds", 0)
        if delay and delay > 0:
            await asyncio.sleep(delay)

        resp = await self._get_throttled(settings.api_base_url, params=params)
        feed = feedparser.parse(resp.text)

        total = None
        try:
            total = int(getattr(feed.feed, "opensearch_totalresults", None))
        except Exception:
            total = None

        return total, list(feed.entries), search_query

    async def get_by_id(self, arxiv_id: str) -> Optional[Dict[str, Any]]:
        clean_id = arxiv_id.strip().replace("arxiv:", "")
        params = {"id_list": clean_id}

        resp = await self._get_throttled(settings.api_base_url, params=params)
        feed = feedparser.parse(resp.text)
        entries = list(feed.entries)
        if not entries:
            return None
        return entries[0]