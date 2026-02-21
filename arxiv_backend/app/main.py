# arxiv_backend/app/main.py
from __future__ import annotations

import logging

import httpx
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import JSONResponse

from app.config import settings
from app.schemas import ErrorResponse, Paper, SearchRequest, SearchResponse
from app.services.arxiv_client import ArxivClient
from app.services.search_service import SearchService
from app.utils.cache import QueryNormalizer, RequestCoalescer, TTLCache

# --- Logging setup ---
# Ensures your logger.info(...) lines show up in the terminal during uvicorn runs.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("arxiv_backend")

app = FastAPI(title="arXiv Paper Search Backend (MVP)", version="0.1")

# ✅ Stale-if-error support: pass stale window into cache (backward compatible if your TTLCache supports it)
cache = TTLCache(
    ttl_seconds=settings.cache_ttl_seconds,
    stale_if_error_seconds=getattr(settings, "cache_stale_if_error_seconds", 0),
)
coalescer = RequestCoalescer()

arxiv_client = ArxivClient()
service = SearchService(client=arxiv_client)


@app.on_event("shutdown")
async def shutdown_event():
    await arxiv_client.aclose()
    logger.info("Shutdown: ArxivClient closed")


@app.get("/health")
def health():
    return {"status": "ok"}


def _search_cache_key(req: SearchRequest, max_results: int) -> str:
    norm_topic = QueryNormalizer.normalize_topic(req.topic)
    norm_cats = QueryNormalizer.normalize_categories(req.categories)
    return (
        f"q={norm_topic}"
        f"|start={req.start}"
        f"|max={max_results}"
        f"|sort={req.sort_by}:{req.sort_order}"
        f"|cats={norm_cats}"
    )


def _maybe_serve_stale(cache_key: str, response: Response, err: Exception):
    """
    Backward compatible stale-if-error:
    - only works if TTLCache implements get_stale()
    """
    get_stale = getattr(cache, "get_stale", None)
    if callable(get_stale):
        stale = get_stale(cache_key)
        if stale is not None:
            response.headers["X-Cache"] = "STALE"
            response.headers["X-Cache-Key"] = cache_key
            logger.warning(
                "SERVING STALE (upstream failed). cache_key=%s error=%s",
                cache_key,
                str(err),
            )
            return stale
    return None


def _raise_upstream_http_exception(err: Exception) -> None:
    """
    Convert upstream / network failures into correct HTTP status codes.
    """
    # Timeouts: arXiv didn't respond in time
    if isinstance(err, (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.PoolTimeout)):
        raise HTTPException(
            status_code=504,
            detail="Upstream arXiv timed out. Please try again.",
        )

    # Network errors: DNS, connect refused, remote protocol issues, etc.
    if isinstance(err, (httpx.ConnectError, httpx.RemoteProtocolError, httpx.NetworkError)):
        raise HTTPException(
            status_code=502,
            detail="Network error while contacting arXiv. Please try again.",
        )

    # arXiv returned an error status (if you ever call raise_for_status upstream)
    if isinstance(err, httpx.HTTPStatusError):
        code = getattr(err.response, "status_code", 502) or 502
        raise HTTPException(
            status_code=502,
            detail=f"arXiv returned an error (status={code}). Please try again.",
        )

    # Unknown internal error
    raise HTTPException(status_code=500, detail="Internal error in arxiv_backend.")


@app.post(
    "/search",
    response_model=SearchResponse,
    responses={
        400: {"model": ErrorResponse},
        502: {"model": ErrorResponse},
        504: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def search(req: SearchRequest, response: Response):
    max_results = min(req.max_results, settings.max_max_results)
    cache_key = _search_cache_key(req, max_results)

    logger.info("SEARCH request received. cache_key=%s", cache_key)

    # Fresh-only cache hit
    cached = cache.get(cache_key)
    if cached:
        response.headers["X-Cache"] = "HIT"
        response.headers["X-Cache-Key"] = cache_key
        logger.info("CACHE HIT. cache_key=%s", cache_key)
        return cached

    # Cache miss (even if request is later coalesced, this tells you cache state)
    response.headers["X-Cache"] = "MISS"
    response.headers["X-Cache-Key"] = cache_key
    logger.info("CACHE MISS. cache_key=%s", cache_key)

    async def work():
        result = await service.search(
            topic=req.topic,
            start=req.start,
            max_results=max_results,
            sort_by=req.sort_by,
            sort_order=req.sort_order,
            categories=req.categories,
        )

        resp = SearchResponse(
            query=req.topic,
            start=req.start,
            max_results=max_results,
            total_results=result["total"],
            results=result["papers"],
        )

        cache.set(cache_key, resp)
        logger.info("CACHE SET. cache_key=%s", cache_key)
        return resp

    try:
        # Request coalescing here: same cache_key => one upstream request
        return await coalescer.run(cache_key, work)

    except HTTPException:
        # Keep any explicit HTTPExceptions as-is
        raise

    except Exception as e:
        # ✅ Stale-if-error fallback
        stale = _maybe_serve_stale(cache_key, response, e)
        if stale is not None:
            return stale

        logger.exception("Search failed. cache_key=%s error=%s", cache_key, str(e))
        _raise_upstream_http_exception(e)
        # (unreachable, but keeps type-checkers happy)
        raise


@app.get(
    "/paper/{arxiv_id}",
    response_model=Paper,
    responses={
        404: {"model": ErrorResponse},
        502: {"model": ErrorResponse},
        504: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def get_paper(arxiv_id: str, response: Response):
    norm_id = QueryNormalizer.normalize_arxiv_id(arxiv_id)
    cache_key = f"paper={norm_id}"

    logger.info("PAPER request received. cache_key=%s", cache_key)

    # Fresh-only cache hit
    cached = cache.get(cache_key)
    if cached:
        response.headers["X-Cache"] = "HIT"
        response.headers["X-Cache-Key"] = cache_key
        logger.info("CACHE HIT. cache_key=%s", cache_key)
        return cached

    response.headers["X-Cache"] = "MISS"
    response.headers["X-Cache-Key"] = cache_key
    logger.info("CACHE MISS. cache_key=%s", cache_key)

    async def work():
        paper = await service.get_paper(norm_id)
        if not paper:
            raise HTTPException(status_code=404, detail="Paper not found")

        cache.set(cache_key, paper)
        logger.info("CACHE SET. cache_key=%s", cache_key)
        return paper

    try:
        return await coalescer.run(cache_key, work)

    except HTTPException:
        raise

    except Exception as e:
        # ✅ Stale-if-error fallback for paper fetch
        stale = _maybe_serve_stale(cache_key, response, e)
        if stale is not None:
            return stale

        logger.exception("Fetch failed. cache_key=%s error=%s", cache_key, str(e))
        _raise_upstream_http_exception(e)
        raise


@app.exception_handler(HTTPException)
def http_exception_handler(_, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.post("/debug/cache/clear")
async def clear_cache():
    cache.clear()
    logger.info("DEBUG: cache cleared")
    return {"status": "ok", "message": "cache cleared"}