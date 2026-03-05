"""
qa_service.py — RAG-based Q&A over ingested papers.

Flow:
    1. Detect if question is structural (headings, outline, etc.) → use full-doc mode.
    2. Detect if question is about references/bibliography → use references mode.
    3. Embed question with BGE query instruction prefix.
    4. Similarity search in pgvector scoped to the paper's arxiv_id.
    5. (References mode) Also include tail chunks because references are usually at the end.
    6. Build context from retrieved chunks (with a max character cap).
    7. Call GPT-4o with context + question.
    8. Return answer string.
"""

from __future__ import annotations

import logging
import re
from typing import List

from pdf_ingestion.app.paper_store import PaperStore, STATUS_READY
from pdf_ingestion.app.embedder import get_embedder

logger = logging.getLogger(__name__)

# --- Standard RAG config ---
TOP_K = 12
MAX_CONTEXT_CHARS = 12000

# --- Full-doc config (for structural/overview questions) ---
FULL_DOC_TOP_K = 50
FULL_DOC_MAX_CONTEXT_CHARS = 20000

# --- References config (references/bibliography usually live at end of paper) ---
REF_TOP_K = 80
REF_MAX_CONTEXT_CHARS = 22000
REF_TAIL_CHUNKS = 12  # always include the last N chunks for references questions

# Patterns that indicate the user wants a structural/overview answer
# that requires content from across the whole paper, not top-k similarity hits
_STRUCTURAL_PATTERNS = re.compile(
    r"\b("
    r"heading|headings|header|headers|section|sections|"
    r"outline|table of contents|structure|structure of|"
    r"chapter|chapters|topic|topics|covered|"
    r"what does this paper cover|what is covered|"
    r"overview|summary|summarize|summarise|"
    r"main point|main points|key point|key points|"
    r"conclusion|conclusions|finding|findings|result|results|outcome|outcomes|"
    r"contribution|contributions|what did they|what was done|"
    r"methodology|methods|approach|dataset|datasets|experiment|experiments"
    r")\b",
    re.IGNORECASE,
)

# Patterns that indicate the user wants references/bibliography/citations
_REFERENCES_PATTERNS = re.compile(
    r"\b("
    r"references|bibliography|works cited|citations|reference list|cited works"
    r")\b",
    re.IGNORECASE,
)

_QA_SYSTEM_PROMPT = """\
You are a precise research assistant. You are given excerpts from a scientific \
paper and a question about it.

Rules:
- Answer ONLY from the provided excerpts. Do not use outside knowledge.
- If the answer is not in the excerpts, say exactly: \
  "I couldn't find that in the paper. Try asking something else."
- Be concise and factual.
- For mathematical formulas, reproduce them as they appear in the text.
- For tables, present the relevant rows/values clearly in plain text.
- Do NOT speculate or hallucinate.
"""

_REFERENCES_SYSTEM_PROMPT = """\
You are a precise research assistant. You are given excerpts from the END of a scientific paper
and a question about its references/bibliography.

Rules:
- ONLY list references that are explicitly present in the provided text.
- Do NOT invent citations or author names.
- If the references section is not present in the excerpts, say exactly:
  "I couldn't find the full references in the paper. Try asking something else."
- If the user asks to "list" or "show" references and only a partial list is present,
  list ONLY what is present and mention that it appears partial.
- Keep each reference on its own line. Do not add commentary.
"""

_STRUCTURAL_SYSTEM_PROMPT = """\
You are a precise research assistant. You are given the full text of a scientific \
paper (split into chunks) and a question about its structure or content.

Rules:
- Answer from the provided text. Do not use outside knowledge.
- For questions about headings/sections/structure: list every distinct section \
  heading you can find in the text, in order.
- For questions about findings/outcomes/results: summarize what the paper \
  explicitly states.
- For questions about methodology/datasets: describe what is stated in the text.
- Be thorough — the user wants complete coverage, not just the top match.
- Do NOT add any disclaimers, caveats, or notes about missing sections, \
  incomplete excerpts, or content that might exist beyond what is provided. \
  Simply answer based on what is present. If something is not in the text at \
  all, silently omit it — do not mention its absence.
- Do NOT speculate or hallucinate.
"""


class QAService:
    def __init__(self, store: PaperStore, llm_client) -> None:
        self._store = store
        self._llm = llm_client

    def _is_structural_question(self, question: str) -> bool:
        """Detect questions that need full-document context rather than top-k similarity."""
        return bool(_STRUCTURAL_PATTERNS.search(question))

    def _is_references_question(self, question: str) -> bool:
        """Detect questions specifically about references/bibliography/citations."""
        return bool(_REFERENCES_PATTERNS.search(question))

    def answer(self, arxiv_id: str, question: str, paper_title: str = "") -> str:
        """
        Answer a question about an ingested paper.
        Returns a plain-text answer string (never raises).
        """
        arxiv_id = (arxiv_id or "").strip()
        question = (question or "").strip()

        if not arxiv_id:
            return "No paper is currently loaded. Open a paper first with `open <n>`."

        if not question:
            return "Please ask a question about the paper."

        # Verify paper is ready
        status = self._store.get_paper_status(arxiv_id)
        if status != STATUS_READY:
            if status == "processing":
                return "The paper is still being processed. Please try again in a moment."
            return "This paper hasn't been ingested yet. Open it first with `open <n>`."

        # Choose retrieval strategy based on question type
        is_references = self._is_references_question(question)
        is_structural = self._is_structural_question(question) or is_references

        if is_references:
            top_k = REF_TOP_K
            max_chars = REF_MAX_CONTEXT_CHARS
            system_prompt = _REFERENCES_SYSTEM_PROMPT
        else:
            top_k = FULL_DOC_TOP_K if is_structural else TOP_K
            max_chars = FULL_DOC_MAX_CONTEXT_CHARS if is_structural else MAX_CONTEXT_CHARS
            system_prompt = _STRUCTURAL_SYSTEM_PROMPT if is_structural else _QA_SYSTEM_PROMPT

        logger.info(
            "[QA] arxiv_id=%s structural=%s references=%s top_k=%d max_chars=%d question=%r",
            arxiv_id, is_structural, is_references, top_k, max_chars, question,
        )

        # Embed the question
        try:
            query_vec = get_embedder().embed_query(question)
        except Exception as e:
            logger.error("[QA] embed_query failed: %s", e)
            return "Sorry, I had trouble processing your question. Please try again."

        # Retrieve top-k chunks
        try:
            hits = self._store.search_chunks(arxiv_id, query_vec, top_k=top_k)
        except Exception as e:
            logger.error("[QA] search_chunks failed: %s", e)
            return "Sorry, I couldn't search the paper content. Please try again."

        if not hits:
            return "I have no content stored for this paper. Try re-opening it with `open <n>`."

        # (References mode) Always include the last N chunks, because references are usually at the end.
        if is_references:
            tail_hits = []
            if hasattr(self._store, "get_tail_chunks"):
                try:
                    tail_hits = self._store.get_tail_chunks(arxiv_id, n=REF_TAIL_CHUNKS)
                except Exception as e:
                    logger.warning("[QA] get_tail_chunks failed: %s", e)
                    tail_hits = []

            if tail_hits:
                # Merge by chunk_index, keep existing similarity hits, add missing tail chunks
                merged = {ci: (ci, text, score) for ci, text, score in hits}
                for ci, text, score in tail_hits:
                    merged.setdefault(ci, (ci, text, score))
                hits = list(merged.values())

        # For structural questions: sort chunks by chunk_index (document order)
        # so headings and sections appear in reading order, not similarity order.
        if is_structural:
            hits = sorted(hits, key=lambda h: h[0])  # h[0] is chunk_index

        # Build context (cap at max_chars)
        context_parts: List[str] = []
        total = 0
        for _ci, text, _score in hits:
            if total + len(text) > max_chars:
                remaining = max_chars - total
                if remaining > 200:
                    context_parts.append(text[:remaining])
                break
            context_parts.append(text)
            total += len(text)

        context = "\n\n---\n\n".join(context_parts)
        title_line = f"Paper: {paper_title}\n\n" if paper_title else ""

        user_prompt = (
            f"{title_line}"
            f"Excerpts from the paper:\n\n"
            f"{context}\n\n"
            f"---\n\n"
            f"Question: {question}"
        )

        # Call LLM
        if not self._llm.enabled():
            return (
                "LLM not configured. Here are the most relevant excerpts:\n\n"
                + "\n\n---\n\n".join(context_parts[:3])
            )

        try:
            answer = self._llm.chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
            )
            return answer.strip() if answer else "The model returned an empty response."
        except Exception as e:
            logger.error("[QA] LLM call failed: %s", e)
            return "Sorry, the LLM call failed. Please try again."