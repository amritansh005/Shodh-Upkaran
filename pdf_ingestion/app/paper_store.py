"""
paper_store.py — PostgreSQL + pgvector storage for papers and chunks.

Tables
------
papers
    arxiv_id        TEXT  PRIMARY KEY
    title           TEXT
    authors         TEXT
    pdf_url         TEXT
    abstract        TEXT
    published_date  TEXT
    status          TEXT   ('pending' | 'processing' | 'ready' | 'failed')
    pdf_bytes       BYTEA
    total_pages     INTEGER
    used_ocr        BOOLEAN
    error_msg       TEXT
    paper_headings  TEXT   -- JSON: [{level,text,page}, ...]
    created_at      TIMESTAMPTZ
    updated_at      TIMESTAMPTZ

paper_chunks
    id              BIGSERIAL PRIMARY KEY
    arxiv_id        TEXT  (FK → papers)
    chunk_index     INTEGER
    page_num        INTEGER
    text            TEXT
    embedding       vector(1024)   -- BAAI/bge-large-en-v1.5 is 1024-dim
    created_at      TIMESTAMPTZ
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import psycopg2
import psycopg2.extras

logger = logging.getLogger(__name__)

STATUS_PENDING    = "pending"
STATUS_PROCESSING = "processing"
STATUS_READY      = "ready"
STATUS_FAILED     = "failed"


class PaperStore:
    def __init__(self, dsn: str) -> None:
        self._dsn = dsn

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def _connect(self):
        return psycopg2.connect(self._dsn)

    # ------------------------------------------------------------------
    # Schema bootstrap
    # ------------------------------------------------------------------

    def init_db(self) -> None:
        """Create tables and indexes. Safe to call on every startup."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

                cur.execute("""
                    CREATE TABLE IF NOT EXISTS papers (
                        arxiv_id        TEXT PRIMARY KEY,
                        title           TEXT NOT NULL DEFAULT '',
                        authors         TEXT NOT NULL DEFAULT '',
                        pdf_url         TEXT NOT NULL DEFAULT '',
                        abstract        TEXT NOT NULL DEFAULT '',
                        published_date  TEXT NOT NULL DEFAULT '',
                        status          TEXT NOT NULL DEFAULT 'pending',
                        pdf_bytes       BYTEA,
                        total_pages     INTEGER,
                        used_ocr        BOOLEAN DEFAULT FALSE,
                        error_msg       TEXT,
                        paper_headings  TEXT,
                        created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    );
                """)

                # Safe migration for existing databases
                cur.execute("""
                    ALTER TABLE papers
                        ADD COLUMN IF NOT EXISTS paper_headings TEXT;
                """)

                cur.execute("""
                    CREATE TABLE IF NOT EXISTS paper_chunks (
                        id          BIGSERIAL PRIMARY KEY,
                        arxiv_id    TEXT NOT NULL REFERENCES papers(arxiv_id) ON DELETE CASCADE,
                        chunk_index INTEGER NOT NULL,
                        page_num    INTEGER,
                        text        TEXT NOT NULL,
                        embedding   vector(1024),
                        created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    );
                """)

                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_chunks_arxiv_id
                        ON paper_chunks(arxiv_id);
                """)

            conn.commit()
        logger.info("[PaperStore] DB initialised.")

    def ensure_vector_index(self) -> None:
        """Create IVFFlat index for fast cosine similarity search."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_chunks_embedding
                        ON paper_chunks USING ivfflat (embedding vector_cosine_ops)
                        WITH (lists = 50);
                """)
            conn.commit()

    # ------------------------------------------------------------------
    # Paper CRUD
    # ------------------------------------------------------------------

    def get_paper_status(self, arxiv_id: str) -> Optional[str]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT status FROM papers WHERE arxiv_id = %s;",
                    (arxiv_id,),
                )
                row = cur.fetchone()
                return row[0] if row else None

    def upsert_paper_meta(
        self,
        arxiv_id: str,
        title: str,
        authors: str,
        pdf_url: str,
        abstract: str = "",
        published_date: str = "",
        status: str = STATUS_PENDING,
    ) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO papers
                        (arxiv_id, title, authors, pdf_url, abstract, published_date, status, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
                    ON CONFLICT (arxiv_id) DO UPDATE SET
                        title          = EXCLUDED.title,
                        authors        = EXCLUDED.authors,
                        pdf_url        = EXCLUDED.pdf_url,
                        abstract       = EXCLUDED.abstract,
                        published_date = EXCLUDED.published_date,
                        status         = EXCLUDED.status,
                        updated_at     = NOW();
                """, (arxiv_id, title, authors, pdf_url, abstract, published_date, status))
            conn.commit()

    def save_pdf_bytes(self, arxiv_id: str, pdf_bytes: bytes) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE papers
                    SET pdf_bytes = %s, updated_at = NOW()
                    WHERE arxiv_id = %s;
                """, (psycopg2.Binary(pdf_bytes), arxiv_id))
            conn.commit()

    def mark_processing(self, arxiv_id: str) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE papers SET status = %s, updated_at = NOW() WHERE arxiv_id = %s;",
                    (STATUS_PROCESSING, arxiv_id),
                )
            conn.commit()

    def mark_ready(self, arxiv_id: str, total_pages: int, used_ocr: bool) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE papers
                    SET status = %s, total_pages = %s, used_ocr = %s,
                        error_msg = NULL, updated_at = NOW()
                    WHERE arxiv_id = %s;
                """, (STATUS_READY, total_pages, used_ocr, arxiv_id))
            conn.commit()

    def mark_failed(self, arxiv_id: str, error_msg: str) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE papers
                    SET status = %s, error_msg = %s, updated_at = NOW()
                    WHERE arxiv_id = %s;
                """, (STATUS_FAILED, error_msg[:1000], arxiv_id))
            conn.commit()

    # ------------------------------------------------------------------
    # Structural headings
    # ------------------------------------------------------------------

    def save_headings(self, arxiv_id: str, headings_json: str) -> None:
        """Persist the extracted heading tree (JSON string) for a paper."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE papers
                    SET paper_headings = %s, updated_at = NOW()
                    WHERE arxiv_id = %s;
                    """,
                    (headings_json, arxiv_id),
                )
            conn.commit()

    def get_headings(self, arxiv_id: str) -> Optional[str]:
        """Return stored headings JSON, or None if not yet extracted."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT paper_headings FROM papers WHERE arxiv_id = %s;",
                    (arxiv_id,),
                )
                row = cur.fetchone()
                return row[0] if row else None

    # ------------------------------------------------------------------
    # Chunk CRUD
    # ------------------------------------------------------------------


    def chunk_count(self, arxiv_id: str) -> int:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) FROM paper_chunks WHERE arxiv_id = %s;",
                    (arxiv_id,),
                )
                row = cur.fetchone()
                return int(row[0]) if row else 0

    def delete_chunks(self, arxiv_id: str) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM paper_chunks WHERE arxiv_id = %s;",
                    (arxiv_id,),
                )
            conn.commit()

    def insert_chunks(
        self,
        arxiv_id: str,
        chunks: List[Tuple[int, Optional[int], str, List[float]]],
    ) -> None:
        """
        Bulk insert chunks.
        chunks: list of (chunk_index, page_num, text, embedding_vector)
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                psycopg2.extras.execute_values(
                    cur,
                    """
                    INSERT INTO paper_chunks (arxiv_id, chunk_index, page_num, text, embedding)
                    VALUES %s;
                    """,
                    [
                        (arxiv_id, ci, pn, text, embedding)
                        for ci, pn, text, embedding in chunks
                    ],
                    template="(%s, %s, %s, %s, %s::vector)",
                )
            conn.commit()

    def search_chunks(
        self,
        arxiv_id: str,
        query_embedding: List[float],
        top_k: int = 6,
    ) -> List[Tuple[int, str, float]]:
        """
        Cosine similarity search scoped to one paper.
        Returns list of (chunk_index, text, score).
        """
        embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT chunk_index, text,
                           1 - (embedding <=> %s::vector) AS score
                    FROM paper_chunks
                    WHERE arxiv_id = %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s;
                """, (embedding_str, arxiv_id, embedding_str, top_k))
                rows = cur.fetchall()
                return [(r[0], r[1], float(r[2])) for r in rows]

    # ------------------------------------------------------------------
    # NEW: Tail retrieval for references mode
    # ------------------------------------------------------------------

    def get_tail_chunks(
        self,
        arxiv_id: str,
        n: int = 10,
    ) -> List[Tuple[int, str, float]]:
        """
        Fetch the last N chunks of the paper by chunk_index.
        Used for references/bibliography queries (usually at end of paper).

        Returns list of (chunk_index, text, score=0.0),
        sorted in ascending document order.
        """
        n = max(0, int(n))
        if n == 0:
            return []

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT chunk_index, text
                    FROM paper_chunks
                    WHERE arxiv_id = %s
                    ORDER BY chunk_index DESC
                    LIMIT %s;
                    """,
                    (arxiv_id, n),
                )
                rows = cur.fetchall()

                # Reverse to return in correct document order
                rows = list(reversed(rows))
                return [(int(r[0]), r[1], 0.0) for r in rows]