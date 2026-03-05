"""
pdf_downloader.py — Downloads a PDF and classifies failures precisely.

Instead of a generic fallback, every failure returns a specific human-readable
reason so the chatbot can tell the user exactly what went wrong.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": (
        "ShodhUpkaran/1.0 (research assistant; arxiv paper reader) "
        "Mozilla/5.0 (compatible)"
    )
}


@dataclass
class DownloadResult:
    success: bool
    pdf_bytes: Optional[bytes] = None
    error_message: Optional[str] = None   # human-readable reason shown to user


async def download_pdf(pdf_url: str, timeout_s: float = 120.0) -> DownloadResult:
    """
    Download a PDF. Returns DownloadResult with either bytes or a
    specific human-readable error message.

    Error messages are written to be shown directly to the user in the chatbot.
    """
    url = (pdf_url or "").strip()
    if not url:
        return DownloadResult(
            success=False,
            error_message="No PDF URL was found for this paper.",
        )

    try:
        async with httpx.AsyncClient(
            headers=_HEADERS,
            follow_redirects=True,
            timeout=httpx.Timeout(connect=10.0, read=timeout_s, write=10.0, pool=5.0),
        ) as client:
            logger.info("[DOWNLOAD] Starting: %s", url)
            response = await client.get(url)

            # Handle HTTP error codes with specific messages
            if response.status_code == 404:
                return DownloadResult(
                    success=False,
                    error_message=(
                        f"The PDF does not exist at the address obtained from arXiv ({url}). "
                        "It may have been removed or the URL is outdated."
                    ),
                )
            if response.status_code == 403:
                return DownloadResult(
                    success=False,
                    error_message=(
                        "Access to the PDF was denied (HTTP 403). "
                        "arXiv may be rate-limiting requests. Please try again in a moment."
                    ),
                )
            if response.status_code == 429:
                return DownloadResult(
                    success=False,
                    error_message=(
                        "arXiv is rate-limiting downloads right now (HTTP 429 — Too Many Requests). "
                        "Please wait a minute and try again."
                    ),
                )
            if response.status_code >= 500:
                return DownloadResult(
                    success=False,
                    error_message=(
                        f"The arXiv server returned an error (HTTP {response.status_code}). "
                        "The server may be temporarily unavailable. Please try again shortly."
                    ),
                )

            response.raise_for_status()

            # Verify the response is actually a PDF
            content = response.content
            if not content:
                return DownloadResult(
                    success=False,
                    error_message="The server returned an empty file. The PDF may not exist.",
                )

            if content[:4] != b"%PDF":
                # arXiv sometimes redirects to an HTML page instead of a PDF
                return DownloadResult(
                    success=False,
                    error_message=(
                        "The URL did not return a valid PDF file — the server may have "
                        "returned an error page instead. The paper might not have a PDF version available."
                    ),
                )

            logger.info("[DOWNLOAD] Success: %s (%d bytes)", url, len(content))
            return DownloadResult(success=True, pdf_bytes=content)

    except httpx.ConnectError:
        return DownloadResult(
            success=False,
            error_message=(
                "Could not connect to the arXiv server. "
                "Please check your internet connection and try again."
            ),
        )
    except httpx.TimeoutException:
        return DownloadResult(
            success=False,
            error_message=(
                f"The download timed out after {int(timeout_s)} seconds. "
                "The arXiv server may be slow right now. Please try again."
            ),
        )
    except httpx.HTTPStatusError as e:
        return DownloadResult(
            success=False,
            error_message=(
                f"The server returned an unexpected HTTP error ({e.response.status_code}). "
                "Please try again."
            ),
        )
    except Exception as e:
        logger.error("[DOWNLOAD] Unexpected error for %s: %s", url, e)
        return DownloadResult(
            success=False,
            error_message=f"An unexpected error occurred while downloading: {e}",
        )
