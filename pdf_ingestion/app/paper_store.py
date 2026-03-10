"""
paper_store.py — PostgreSQL + pgvector storage for papers and chunks.

Tables
------
papers
    arxiv_id                  TEXT  PRIMARY KEY
    title                     TEXT
    authors                   TEXT
    pdf_url                   TEXT
    abstract                  TEXT
    published_date            TEXT
    status                    TEXT   ('pending' | 'processing' | 'ready' | 'failed')
    pdf_bytes                 BYTEA
    total_pages               INTEGER
    used_ocr                  BOOLEAN
    error_msg                 TEXT
    paper_headings            TEXT   -- JSON: [{level,text,page}, ...]
    reference_heading         TEXT
    reference_start_page      INTEGER
    reference_end_page        INTEGER
    reference_count           INTEGER
    last_reference_number     INTEGER
    reference_numbering_style TEXT
    created_at                TIMESTAMPTZ
    updated_at                TIMESTAMPTZ

paper_chunks
    id              BIGSERIAL PRIMARY KEY
    arxiv_id        TEXT  (FK → papers)
    chunk_index     INTEGER
    page_num        INTEGER
    text            TEXT
    embedding       vector(1024)   -- BAAI/bge-large-en-v1.5 is 1024-dim
    section_heading TEXT           -- heading this chunk belongs to, or NULL
    created_at      TIMESTAMPTZ

paper_sections
    id              BIGSERIAL PRIMARY KEY
    arxiv_id        TEXT  (FK → papers)
    section_index   INTEGER        -- 0-based order in document
    heading_level   INTEGER        -- 0=preamble, 1=top, 2=subsection, 3=sub-sub
    heading_text    TEXT           -- exact heading as it appears
    parent_heading  TEXT           -- nearest parent heading text, or NULL
    page_start      INTEGER        -- 1-based page where section begins
    page_end        INTEGER        -- 1-based page where section ends (inclusive)
    content_text    TEXT           -- full extracted text for this section
    content_length  INTEGER        -- character count
    created_at      TIMESTAMPTZ
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import psycopg2
import psycopg2.extras

logger = logging.getLogger(__name__)

STATUS_PENDING = "pending"
STATUS_PROCESSING = "processing"
STATUS_READY = "ready"
STATUS_FAILED = "failed"


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
                        arxiv_id                  TEXT PRIMARY KEY,
                        title                     TEXT NOT NULL DEFAULT '',
                        authors                   TEXT NOT NULL DEFAULT '',
                        pdf_url                   TEXT NOT NULL DEFAULT '',
                        abstract                  TEXT NOT NULL DEFAULT '',
                        published_date            TEXT NOT NULL DEFAULT '',
                        status                    TEXT NOT NULL DEFAULT 'pending',
                        pdf_bytes                 BYTEA,
                        total_pages               INTEGER,
                        used_ocr                  BOOLEAN DEFAULT FALSE,
                        error_msg                 TEXT,
                        paper_headings            TEXT,
                        reference_heading         TEXT,
                        reference_start_page      INTEGER,
                        reference_end_page        INTEGER,
                        reference_count           INTEGER,
                        last_reference_number     INTEGER,
                        reference_numbering_style TEXT,
                        created_at                TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        updated_at                TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    );
                """)

                # Safe migrations for existing databases
                cur.execute("""
                    ALTER TABLE papers
                        ADD COLUMN IF NOT EXISTS paper_headings TEXT,
                        ADD COLUMN IF NOT EXISTS reference_heading TEXT,
                        ADD COLUMN IF NOT EXISTS reference_start_page INTEGER,
                        ADD COLUMN IF NOT EXISTS reference_end_page INTEGER,
                        ADD COLUMN IF NOT EXISTS reference_count INTEGER,
                        ADD COLUMN IF NOT EXISTS last_reference_number INTEGER,
                        ADD COLUMN IF NOT EXISTS reference_numbering_style TEXT;
                """)

                cur.execute("""
                    CREATE TABLE IF NOT EXISTS paper_chunks (
                        id               BIGSERIAL PRIMARY KEY,
                        arxiv_id         TEXT NOT NULL REFERENCES papers(arxiv_id) ON DELETE CASCADE,
                        chunk_index      INTEGER NOT NULL,
                        page_num         INTEGER,
                        text             TEXT NOT NULL,
                        embedding        vector(1024),
                        section_heading  TEXT,
                        created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    );
                """)

                # Safe migration for existing databases
                cur.execute("""
                    ALTER TABLE paper_chunks
                        ADD COLUMN IF NOT EXISTS section_heading TEXT;
                """)

                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_chunks_arxiv_id
                        ON paper_chunks(arxiv_id);
                """)

                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_chunks_section_heading
                        ON paper_chunks(arxiv_id, section_heading)
                        WHERE section_heading IS NOT NULL;
                """)

                # ── paper_sections table ──────────────────────────────────────
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS paper_sections (
                        id              BIGSERIAL PRIMARY KEY,
                        arxiv_id        TEXT NOT NULL REFERENCES papers(arxiv_id) ON DELETE CASCADE,
                        section_index   INTEGER NOT NULL,
                        heading_level   INTEGER NOT NULL DEFAULT 1,
                        heading_text    TEXT NOT NULL,
                        parent_heading  TEXT,
                        page_start      INTEGER,
                        page_end        INTEGER,
                        content_text    TEXT NOT NULL DEFAULT '',
                        content_length  INTEGER NOT NULL DEFAULT 0,
                        created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    );
                """)

                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_sections_arxiv_id
                        ON paper_sections(arxiv_id);
                """)

                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_sections_order
                        ON paper_sections(arxiv_id, section_index);
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
    # Reference metadata
    # ------------------------------------------------------------------

    def save_reference_metadata(
        self,
        arxiv_id: str,
        reference_heading: Optional[str],
        reference_start_page: Optional[int],
        reference_end_page: Optional[int],
        reference_count: Optional[int],
        last_reference_number: Optional[int],
        reference_numbering_style: Optional[str],
    ) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE papers
                    SET
                        reference_heading = %s,
                        reference_start_page = %s,
                        reference_end_page = %s,
                        reference_count = %s,
                        last_reference_number = %s,
                        reference_numbering_style = %s,
                        updated_at = NOW()
                    WHERE arxiv_id = %s;
                """, (
                    reference_heading,
                    reference_start_page,
                    reference_end_page,
                    reference_count,
                    last_reference_number,
                    reference_numbering_style,
                    arxiv_id,
                ))
            conn.commit()

    def get_reference_metadata(self, arxiv_id: str) -> Optional[Dict[str, Optional[object]]]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT
                        reference_heading,
                        reference_start_page,
                        reference_end_page,
                        reference_count,
                        last_reference_number,
                        reference_numbering_style
                    FROM papers
                    WHERE arxiv_id = %s;
                """, (arxiv_id,))
                row = cur.fetchone()

                if not row:
                    return None

                return {
                    "reference_heading": row[0],
                    "reference_start_page": row[1],
                    "reference_end_page": row[2],
                    "reference_count": row[3],
                    "last_reference_number": row[4],
                    "reference_numbering_style": row[5],
                }

    def clear_reference_metadata(self, arxiv_id: str) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE papers
                    SET
                        reference_heading = NULL,
                        reference_start_page = NULL,
                        reference_end_page = NULL,
                        reference_count = NULL,
                        last_reference_number = NULL,
                        reference_numbering_style = NULL,
                        updated_at = NOW()
                    WHERE arxiv_id = %s;
                """, (arxiv_id,))
            conn.commit()

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
        chunks: List[Tuple],
    ) -> None:
        """
        Bulk insert chunks.
        chunks: list of (chunk_index, page_num, text, embedding_vector, section_heading)
        section_heading may be None for chunks produced by the page-level fallback.
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                psycopg2.extras.execute_values(
                    cur,
                    """
                    INSERT INTO paper_chunks
                        (arxiv_id, chunk_index, page_num, text, embedding, section_heading)
                    VALUES %s;
                    """,
                    [
                        (arxiv_id, ci, pn, text, embedding, section_heading)
                        for ci, pn, text, embedding, section_heading in chunks
                    ],
                    template="(%s, %s, %s, %s, %s::vector, %s)",
                )
            conn.commit()

    def search_chunks(
        self,
        arxiv_id: str,
        query_embedding: List[float],
        top_k: Optional[int] = 6,
        section_heading: Optional[str] = None,
    ) -> List[Tuple[int, str, float, Optional[str]]]:
        """
        Cosine similarity search scoped to one paper.
        Returns list of (chunk_index, text, score, section_heading).

        If section_heading is provided, restricts search to chunks from
        that section only (case-insensitive ILIKE match).

        If top_k is None, returns ALL matching chunks with no LIMIT —
        used when a section filter is active so every chunk from that
        section is returned regardless of how many there are.
        """
        embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

        with self._connect() as conn:
            with conn.cursor() as cur:
                if section_heading and top_k is None:
                    cur.execute("""
                        SELECT chunk_index, text,
                               1 - (embedding <=> %s::vector) AS score,
                               section_heading
                        FROM paper_chunks
                        WHERE arxiv_id = %s
                          AND section_heading ILIKE %s
                        ORDER BY chunk_index ASC;
                    """, (embedding_str, arxiv_id, f"%{section_heading}%"))
                elif section_heading:
                    cur.execute("""
                        SELECT chunk_index, text,
                               1 - (embedding <=> %s::vector) AS score,
                               section_heading
                        FROM paper_chunks
                        WHERE arxiv_id = %s
                          AND section_heading ILIKE %s
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s;
                    """, (embedding_str, arxiv_id, f"%{section_heading}%", embedding_str, top_k))
                else:
                    cur.execute("""
                        SELECT chunk_index, text,
                               1 - (embedding <=> %s::vector) AS score,
                               section_heading
                        FROM paper_chunks
                        WHERE arxiv_id = %s
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s;
                    """, (embedding_str, arxiv_id, embedding_str, top_k))
                rows = cur.fetchall()
                return [(r[0], r[1], float(r[2]), r[3]) for r in rows]

    # ------------------------------------------------------------------
    # Tail retrieval for references mode
    # ------------------------------------------------------------------

    def get_tail_chunks(
        self,
        arxiv_id: str,
        n: int = 10,
    ) -> List[Tuple[int, str, float, Optional[str]]]:
        """
        Fetch the last N chunks of the paper by chunk_index.
        Used for references/bibliography queries (usually at end of paper).

        Returns list of (chunk_index, text, score=0.0, section_heading),
        sorted in ascending document order.
        """
        n = max(0, int(n))
        if n == 0:
            return []

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT chunk_index, text, section_heading
                    FROM paper_chunks
                    WHERE arxiv_id = %s
                    ORDER BY chunk_index DESC
                    LIMIT %s;
                    """,
                    (arxiv_id, n),
                )
                rows = cur.fetchall()
                rows = list(reversed(rows))
                return [(int(r[0]), r[1], 0.0, r[2]) for r in rows]

    # ------------------------------------------------------------------
    # Section CRUD
    # ------------------------------------------------------------------

    def delete_sections(self, arxiv_id: str) -> None:
        """Delete all sections for a paper (called before re-ingesting)."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM paper_sections WHERE arxiv_id = %s;",
                    (arxiv_id,),
                )
            conn.commit()

    def insert_sections(
        self,
        arxiv_id: str,
        sections: list,
    ) -> None:
        """
        Bulk insert all sections for a paper.
        sections: List[PaperSection] from section_assembler.assemble_sections()
        """
        if not sections:
            return

        with self._connect() as conn:
            with conn.cursor() as cur:
                psycopg2.extras.execute_values(
                    cur,
                    """
                    INSERT INTO paper_sections
                        (arxiv_id, section_index, heading_level, heading_text,
                         parent_heading, page_start, page_end,
                         content_text, content_length)
                    VALUES %s;
                    """,
                    [
                        (
                            arxiv_id,
                            s.section_index,
                            s.heading_level,
                            s.heading_text,
                            s.parent_heading,
                            s.page_start,
                            s.page_end,
                            s.content_text,
                            s.content_length,
                        )
                        for s in sections
                    ],
                )
            conn.commit()
        logger.info("[PaperStore] Inserted %d sections for %s.", len(sections), arxiv_id)

    def get_sections(self, arxiv_id: str) -> list:
        """
        Return all sections for a paper in document order.
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT section_index, heading_level, heading_text,
                           parent_heading, page_start, page_end,
                           content_text, content_length
                    FROM paper_sections
                    WHERE arxiv_id = %s
                    ORDER BY section_index ASC;
                    """,
                    (arxiv_id,),
                )
                rows = cur.fetchall()
                return [
                    {
                        "section_index": r[0],
                        "heading_level": r[1],
                        "heading_text": r[2],
                        "parent_heading": r[3],
                        "page_start": r[4],
                        "page_end": r[5],
                        "content_text": r[6],
                        "content_length": r[7],
                    }
                    for r in rows
                ]

    def section_count(self, arxiv_id: str) -> int:
        """Return how many sections are stored for a paper."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) FROM paper_sections WHERE arxiv_id = %s;",
                    (arxiv_id,),
                )
                row = cur.fetchone()
                return int(row[0]) if row else 0

    def get_section_by_heading(
        self,
        arxiv_id: str,
        heading_query: str,
    ) -> Optional[dict]:
        """
        Find a section by approximate heading match (case-insensitive ILIKE).
        Returns the best matching section dict, or None.
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT section_index, heading_level, heading_text,
                           parent_heading, page_start, page_end,
                           content_text, content_length
                    FROM paper_sections
                    WHERE arxiv_id = %s
                      AND heading_text ILIKE %s
                    ORDER BY section_index ASC
                    LIMIT 1;
                    """,
                    (arxiv_id, f"%{heading_query}%"),
                )
                row = cur.fetchone()
                if not row:
                    return None

                return {
                    "section_index": row[0],
                    "heading_level": row[1],
                    "heading_text": row[2],
                    "parent_heading": row[3],
                    "page_start": row[4],
                    "page_end": row[5],
                    "content_text": row[6],
                    "content_length": row[7],
                }