from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ARXIV_",
        env_file=".env",
        extra="ignore",
    )

    # arXiv API endpoint (Atom feed)
    api_base_url: str = "https://export.arxiv.org/api/query"

    # Timeouts & caching
    http_timeout_seconds: float = 10.0

    # Fresh cache TTL (normal behavior)
    cache_ttl_seconds: int = 600  # 10 minutes

    # ---------------------------------------------
    # Stale-if-error cache window
    # Used ONLY if upstream arXiv fails (429/5xx/timeout)
    # Effective max age ≈ cache_ttl_seconds + this value
    # ---------------------------------------------
    cache_stale_if_error_seconds: int = 3600  # 1 hour

    # Default search behavior
    default_max_results: int = 10
    max_max_results: int = 500  # hard cap to protect your backend

    # User agent (arXiv asks for reasonable identification)
    user_agent: str = "arxiv-backend-mvp/0.1 (contact: you@example.com)"

    # -------------------------------------------------
    # DEV ONLY: Artificial delay to test request coalescing
    # Set to 2 (or higher) while testing.
    # Set to 0 in normal / production usage.
    # Can be overridden via env: ARXIV_COALESCE_TEST_DELAY_SECONDS
    # -------------------------------------------------
    coalesce_test_delay_seconds: int = 2

    # -------------------------------------------------
    # GLOBAL outgoing throttle (client-side)
    # Applies to ALL outgoing requests to arXiv
    # Can be overridden via env:
    #   ARXIV_OUTGOING_MAX_CONCURRENCY
    #   ARXIV_OUTGOING_RPS
    #   ARXIV_OUTGOING_BURST
    #   ARXIV_OUTGOING_MAX_RETRIES
    #   ARXIV_OUTGOING_RETRY_BACKOFF_BASE_SECONDS
    # -------------------------------------------------
    outgoing_max_concurrency: int = 5   # max parallel upstream calls
    outgoing_rps: float = 2.0           # sustained requests/sec
    outgoing_burst: int = 5             # burst tokens (short spikes)

    # Retries for upstream throttles/transient failures
    outgoing_max_retries: int = 3
    outgoing_retry_backoff_base_seconds: float = 0.5  # exponential backoff base


settings = Settings()
