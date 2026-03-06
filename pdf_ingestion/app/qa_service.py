"""
qa_service.py — RAG-based Q&A over ingested papers.

Flow:
    1. Ask GPT-4o to classify the question as structural / references / specific.
    2. Embed question with BGE query instruction prefix.
    3. Similarity search in pgvector scoped to the paper's arxiv_id.
    4. (References mode) Also include tail chunks because references are usually at the end.
    5. Build context from retrieved chunks (with a max character cap).
    6. Call GPT-4o with context + question.
    7. Return answer string.
"""

from __future__ import annotations

import logging
from typing import List, Literal

from pdf_ingestion.app.paper_store import PaperStore, STATUS_READY
from pdf_ingestion.app.embedder import get_embedder
from pdf_ingestion.app.structural_extractor import HeadingTree

logger = logging.getLogger(__name__)

# --- Standard RAG config ---
TOP_K = 12
MAX_CONTEXT_CHARS = 12000

# --- Full-doc config (for structural/overview questions) ---
FULL_DOC_TOP_K = 50
FULL_DOC_MAX_CONTEXT_CHARS = 20000

# --- References config (references/bibliography usually live at end of paper) ---
REF_TOP_K = 80
REF_MAX_CONTEXT_CHARS = 40000  # references lists are long — give them enough room
REF_TAIL_CHUNKS = 30  # last 30 chunks covers ~12 pages, enough for any references section

# Question type labels returned by GPT-4o classifier
QuestionType = Literal["structural", "references", "specific"]

# Lightweight hard overrides (guardrails) to reduce misrouting
# NOTE: These do NOT attempt to enumerate paper headings; they only detect question intent.
_HARD_REFERENCES_KEYWORDS = (
    "references",
    "bibliography",
    "works cited",
    "citations",
    "reference list",
    "cited works",
)

_HARD_STRUCTURAL_KEYWORDS = (
    "heading",
    "headings",
    "header",
    "headers",
    "outline",
    "table of contents",
    "toc",
    "sections",
    "structure",
    "chapter",
    "chapters",
)

# Keywords that specifically mean "list the headings/structure" —
# for these we answer directly from the stored HeadingTree, no embeddings needed.
_HEADING_QUERY_KEYWORDS = (
    "heading",
    "headings",
    "header",
    "headers",
    "outline",
    "table of contents",
    "toc",
)

_HEADING_SYSTEM_PROMPT = """\
You are a precise research assistant. You are given the complete section
heading outline of a research paper, extracted visually from page images.

Rules:
- List the headings EXACTLY as provided — do not add, remove, or rename any.
- Use a bullet list with dashes (-).
- Indent sub-headings (level 2) with two spaces, sub-sub-headings (level 3) with four.
- Include the page number in parentheses at the end of each line,
  e.g.:  - I. Introduction  (page 1)
- Do NOT add any commentary, caveats, or extra text beyond the list.
"""

_CLASSIFIER_SYSTEM_PROMPT = """\
You are a router for a research-paper Q&A system.

Classify the user's question into EXACTLY ONE label:

structural:
- asks for headings/sections/outline/table of contents/structure
- asks for overview/summary/key takeaways/main points/contributions
- asks to explain or summarize a SECTION or broad topic, e.g.:
  applications, use cases, implications, limitations, discussion, future work,
  conclusion, results, findings, methodology, approach, datasets, experiments
- asks for a brief description like "in 2–3 lines" about a broad part of the paper

references:
- asks for references/bibliography/citations/works cited/reference list

specific:
- asks for a single narrow fact/value/definition/equation/detail from a small passage

Output rules:
- Reply with ONLY one of these three words: structural, references, specific
- No punctuation. No extra words. No explanation.
"""

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
You are a precise research assistant. You are given excerpts from a scientific paper
and a question about its references/bibliography.

Rules:
- ONLY list references that are explicitly present in the provided text.
- Copy each reference EXACTLY and VERBATIM as it appears in the text — do NOT paraphrase,
  summarize, shorten, or rewrite any reference in any way.
- Do NOT merge multiple references into a single line. Each reference must be on its own line.
- Do NOT invent citations, author names, titles, or any other detail.
- Do NOT add any commentary, annotations, or descriptions alongside the references.
- If the excerpts contain ANY references at all, list every one of them verbatim.
- If only a partial list is present, list what is there verbatim, then add one line:
  "Note: This list may be incomplete — not all references were found in the retrieved excerpts."
- ONLY say "I couldn't find the references in the paper." if there are genuinely zero
  reference entries anywhere in the provided text. Do not say this if even one reference
  is present.
"""

_STRUCTURAL_SYSTEM_PROMPT = """\
You are a precise research assistant. You are given the full text of a scientific \
paper (split into chunks) and a question about its structure or content.

Rules:
- Answer from the provided text. Do not use outside knowledge.
- For questions about headings/sections/structure: list ONLY genuine section and \
  subsection headings — these are the titled divisions of the paper such as \
  Abstract, Introduction, Related Work, Methodology, Results, Discussion, \
  Conclusion, Future Work, and any numbered or named sub-sections. \
  Use a BULLET LIST (dash "-") — do NOT use a numbered list. \
  Reproduce each heading EXACTLY as it appears in the text \
  (including any section number already in the heading, e.g. \
  "1 Introduction", "4.1 LIME"). Never add your own numbering on top.
- Do NOT include figure captions (e.g. "Figure 1. ..."), table titles \
  (e.g. "Table 2. ..."), or any other non-heading labels in the list.
- For sub-sections, indent them with two extra spaces under their parent section.
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

    def _hard_route(self, question: str) -> QuestionType | None:
        """
        Cheap, deterministic routing for obvious cases.
        Returns a QuestionType if confidently matched, else None.
        """
        q = (question or "").strip().lower()
        if not q:
            return None

        # References overrides structural if both appear.
        if any(k in q for k in _HARD_REFERENCES_KEYWORDS):
            return "references"

        if any(k in q for k in _HARD_STRUCTURAL_KEYWORDS):
            return "structural"

        return None

    def _classify_question(self, question: str) -> QuestionType:
        """
        Ask GPT-4o to classify the question as one of:
          'structural' | 'references' | 'specific'

        Falls back to 'specific' if the LLM is unavailable or returns
        an unexpected label, so retrieval always continues.
        """

        # Hard routing guardrails first (fast + stable)
        hard = self._hard_route(question)
        if hard is not None:
            logger.info("[QA] Hard-routed question as '%s': %r", hard, question)
            return hard

        # If LLM isn't available, keep behavior stable and proceed with specific mode.
        if not self._llm.enabled():
            logger.warning("[QA] LLM not available for classification, defaulting to 'specific'")
            return "specific"

        try:
            raw = self._llm.chat(
                messages=[
                    {"role": "system", "content": _CLASSIFIER_SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                ],
                temperature=0.0,
            )

            # Robust normalization: strip whitespace + common punctuation
            label = (raw or "").strip().lower().strip(" \t\r\n.:-\"'`")

            if label in ("structural", "references", "specific"):
                logger.info("[QA] GPT-4o classified question as '%s': %r", label, question)
                return label  # type: ignore[return-value]

            logger.warning("[QA] Unexpected classification label %r, defaulting to 'specific'", raw)
        except Exception as e:
            logger.error("[QA] Classification call failed: %s — defaulting to 'specific'", e)

        return "specific"

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

        # ── Fast path: heading queries answered from stored HeadingTree ───────
        # At ingest time, GPT-4o vision read every page as an image and
        # extracted the heading structure. For queries asking "what are the
        # headings / outline / table of contents", we skip embedding retrieval
        # entirely and answer directly from that stored data.
        _q = question.lower()
        if any(k in _q for k in _HEADING_QUERY_KEYWORDS):
            stored_json = self._store.get_headings(arxiv_id)
            if stored_json:
                tree = HeadingTree.from_json(stored_json)
                if not tree.is_empty():
                    logger.info(
                        "[QA] Heading fast-path: %d headings stored for %s",
                        len(tree.headings), arxiv_id,
                    )
                    heading_outline = tree.format_for_llm()
                    if not self._llm.enabled():
                        return heading_outline
                    try:
                        answer = self._llm.chat(
                            messages=[
                                {"role": "system", "content": _HEADING_SYSTEM_PROMPT},
                                {
                                    "role": "user",
                                    "content": (
                                        (f"Paper: {paper_title}\n\n" if paper_title else "")
                                        + f"Heading outline:\n{heading_outline}\n\n"
                                        + f"Question: {question}"
                                    ),
                                },
                            ],
                            temperature=0.0,
                        )
                        return answer.strip() if answer else heading_outline
                    except Exception as _e:
                        logger.error("[QA] Heading fast-path LLM call failed: %s", _e)
                        return heading_outline

        # Choose retrieval strategy based on GPT-4o question classification
        question_type = self._classify_question(question)
        is_references = question_type == "references"
        is_structural = question_type in ("structural", "references")

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
                # Merge by chunk_index, deduplicate
                merged = {ci: (ci, text, score) for ci, text, score in hits}
                for ci, text, score in tail_hits:
                    merged.setdefault(ci, (ci, text, score))
                # Sort tail chunks FIRST (ascending document order), then similarity
                # hits from the rest of the paper after. This guarantees references
                # at the end of the paper are never cut off by the MAX_CONTEXT_CHARS
                # cap — they go into context before chunks from the middle of the paper.
                tail_indices = {ci for ci, _, _ in tail_hits}
                tail_part = sorted(
                    [v for v in merged.values() if v[0] in tail_indices],
                    key=lambda h: h[0],   # ascending document order within tail
                )
                other_part = sorted(
                    [v for v in merged.values() if v[0] not in tail_indices],
                    key=lambda h: -h[2],  # descending similarity score
                )
                hits = tail_part + other_part

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
