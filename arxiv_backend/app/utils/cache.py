import asyncio
import logging
import re
import time
import threading
from typing import Any, Dict, Optional, Tuple, Callable, Awaitable, List

# Use the same logger namespace you configured in main.py (logging.basicConfig(...))
logger = logging.getLogger("arxiv_backend")


class TTLCache:
    """
    Tiny in-memory cache with:
      - Fresh TTL (normal HITs)
      - Optional stale-if-error window (fallback only when upstream fails)

    Storage:
      Key -> (fresh_expires_at, stale_expires_at, value)

    Semantics:
      - get(key): returns value only if still FRESH
      - get_stale(key): returns value if within STALE window (fresh or stale)
      - Entries are removed only after stale_expires_at passes
    """
    def __init__(self, ttl_seconds: int = 600, stale_if_error_seconds: int = 0):
        self.ttl_seconds = int(ttl_seconds)
        self.stale_if_error_seconds = int(stale_if_error_seconds)
        self._store: Dict[str, Tuple[float, float, Any]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        """Fresh-only read."""
        now = time.time()
        with self._lock:
            item = self._store.get(key)
            if not item:
                return None

            fresh_expires_at, stale_expires_at, value = item

            # If we're past the stale window, delete and miss
            if now > stale_expires_at:
                self._store.pop(key, None)
                return None

            # Fresh hit
            if now <= fresh_expires_at:
                return value

            # Past fresh TTL but still within stale window -> not fresh
            return None

    def get_stale(self, key: str) -> Optional[Any]:
        """
        Stale-allowed read.
        Use this ONLY when upstream fails (stale-if-error).
        Returns value if within stale window (fresh OR stale).
        """
        now = time.time()
        with self._lock:
            item = self._store.get(key)
            if not item:
                return None

            _, stale_expires_at, value = item

            if now > stale_expires_at:
                self._store.pop(key, None)
                return None

            return value

    def set(self, key: str, value: Any) -> None:
        now = time.time()
        fresh_expires_at = now + self.ttl_seconds
        stale_expires_at = fresh_expires_at + max(0, self.stale_if_error_seconds)
        with self._lock:
            self._store[key] = (fresh_expires_at, stale_expires_at, value)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()


class QueryNormalizer:
    """
    Normalizes user query + category filters so semantically identical requests map to same cache key.
    """
    _space_re = re.compile(r"\s+")

    @classmethod
    def normalize_topic(cls, topic: str) -> str:
        # lower, trim, collapse spaces, remove surrounding quotes-ish noise
        t = (topic or "").strip().lower()
        t = t.replace('"', "").replace("'", "")
        t = cls._space_re.sub(" ", t)
        return t

    @classmethod
    def normalize_categories(cls, categories: Optional[List[str]]) -> str:
        if not categories:
            return ""
        cleaned = []
        for c in categories:
            cc = (c or "").strip()
            if cc:
                cleaned.append(cc)
        # sort so order doesn't matter in cache key
        cleaned = sorted(set(cleaned))
        return ",".join(cleaned)

    @classmethod
    
    @classmethod
    def normalize_author(cls, author: Optional[str]) -> str:
        a = (author or "").strip().lower()
        a = a.replace('"', "").replace("'", "")
        a = cls._space_re.sub(" ", a)
        return a

    @classmethod
    def normalize_year_range(cls, from_year: Optional[int], to_year: Optional[int]) -> str:
        fy = "" if from_year is None else str(int(from_year))
        ty = "" if to_year is None else str(int(to_year))
        if not fy and not ty:
            return ""
        return f"{fy}-{ty}"

def normalize_arxiv_id(cls, arxiv_id: str) -> str:
        # accept "arxiv:xxxx", "xxxxv2", "http://arxiv.org/abs/xxxxv2"
        s = (arxiv_id or "").strip().lower()

        if "/abs/" in s:
            s = s.split("/abs/")[-1]
        if s.startswith("arxiv:"):
            s = s.replace("arxiv:", "", 1)

        # remove version suffix vN if present
        if "v" in s and s.split("v")[-1].isdigit():
            s = s.split("v")[0]
        return s.strip()


class RequestCoalescer:
    """
    Request coalescing / single-flight:
    - If multiple callers request the same key concurrently, run the underlying coroutine once.
    - Others await the same Future.
    """
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._inflight: Dict[str, asyncio.Future] = {}

    async def run(self, key: str, work: Callable[[], Awaitable[Any]]) -> Any:
        fut: Optional[asyncio.Future] = None
        leader = False

        # Decide leader vs waiter under the lock (but DO NOT await under the lock)
        async with self._lock:
            fut = self._inflight.get(key)
            if fut is not None:
                logger.info("COALESCER WAITER. key=%s", key)
            else:
                loop = asyncio.get_running_loop()
                fut = loop.create_future()
                self._inflight[key] = fut
                leader = True
                logger.info("COALESCER LEADER. key=%s", key)

        # Waiter path: just await the in-flight future (outside the lock)
        if not leader:
            return await fut  # type: ignore[arg-type]

        # Leader path: execute the work once, then resolve the future for all waiters
        try:
            result = await work()
            if not fut.done():  # type: ignore[union-attr]
                fut.set_result(result)  # type: ignore[union-attr]
            return result
        except Exception as e:
            if not fut.done():  # type: ignore[union-attr]
                fut.set_exception(e)  # type: ignore[union-attr]
            raise
        finally:
            # Clean up inflight entry so future requests can run again
            async with self._lock:
                self._inflight.pop(key, None)


# -------------------------------------------------------------------
# Outgoing traffic throttle helpers (client-side rate + concurrency)
# -------------------------------------------------------------------

class AsyncTokenBucketRateLimiter:
    """
    Async token-bucket rate limiter:
      - refill at `rate` tokens/sec
      - bucket capacity = `capacity` (burst)
      - acquire() waits until 1 token is available
    """
    def __init__(self, rate: float, capacity: int):
        if rate <= 0:
            raise ValueError("rate must be > 0")
        if capacity <= 0:
            raise ValueError("capacity must be > 0")

        self.rate = float(rate)
        self.capacity = int(capacity)

        self._tokens: float = float(capacity)
        self._last: float = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        while True:
            wait_s: Optional[float] = None

            async with self._lock:
                now = time.monotonic()
                elapsed = now - self._last
                self._last = now

                # refill tokens
                self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)

                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return

                missing = 1.0 - self._tokens
                wait_s = missing / self.rate

            # sleep outside lock
            await asyncio.sleep(wait_s)


class AsyncConcurrencyLimiter:
    """
    Async concurrency limiter using a semaphore.
    Use:
      async with limiter:
          ...
    """
    def __init__(self, max_concurrency: int):
        if max_concurrency <= 0:
            raise ValueError("max_concurrency must be > 0")
        self._sem = asyncio.Semaphore(max_concurrency)

    async def __aenter__(self):
        await self._sem.acquire()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self._sem.release()
        return False
