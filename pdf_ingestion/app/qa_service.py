"""
qa_service.py — RAG-based Q&A over ingested papers.

Flow:
    1. Ask GPT-4o to classify the question into one of four labels:
         outline    — wants the document structure/headings/sections list
         structural — wants content from a broad section or overview
         references — wants the bibliography/citations
         specific   — wants a narrow fact or detail
    2. outline questions are answered directly from the stored HeadingTree
       (no embedding retrieval needed) — this is the "fast path".
    3. For all other types: embed question → pgvector similarity search →
       build context → call GPT-4o with context + question.
    4. (references mode) Also append tail chunks (end of paper).
    5. Simple reference metadata questions (count / last number / invalid
       reference number) are answered directly from stored DB metadata.
"""

from __future__ import annotations

import logging
import re
from typing import List, Literal, Optional

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

# --- Section-filtered config (all chunks from matched section, no top_k cap) ---
# When a question targets a specific named section (e.g. "what does methodology say?"),
# we fetch ALL chunks from that section and pass them ALL to GPT-4o in document order.
# Cap raised to 80k chars (~20k tokens) — covers even the longest Results/Discussion
# sections. GPT-4o supports 128k context so this is safely within limits.
SECTION_MAX_CONTEXT_CHARS = 80000

# --- References config (references/bibliography usually live at end of paper) ---
REF_TOP_K = 80
REF_MAX_CONTEXT_CHARS = 40000  # references lists are long — give them enough room
REF_TAIL_CHUNKS = 30  # last 30 chunks covers ~12 pages, enough for any references section

QuestionType = Literal["outline", "structural", "references", "specific"]

_HARD_REFERENCES_KEYWORDS = (
    "references",
    "bibliography",
    "works cited",
    "citations",
    "reference list",
    "cited works",
)

_GENERIC_SECTION_TRIGGER_MAP = (
    ("abstract", "abstract"),
    ("introduction", "introduction"),
    ("related work", "related work"),
    ("literature review", "related work"),
    ("background", "background"),
    ("methodology", "methodology"),
    ("method", "method"),
    ("approach", "approach"),
    ("experiment", "experiment"),
    ("evaluation", "evaluation"),
    ("results", "result"),
    ("findings", "result"),
    ("discussion", "discussion"),
    ("conclusion", "conclusion"),
    ("future work", "future"),
    ("limitation", "limitation"),
    ("dataset", "dataset"),
    ("implementation", "implementation"),
    ("architecture", "architecture"),
    ("contribution", "contribution"),
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

outline:
- asks what sections, headings, chapters, or topics the paper contains
- asks for the paper's structure, table of contents, or outline
- asks to list/show/enumerate the sections or headings
- Examples: "what sections does this paper have?", "show me the headings",
  "what are the different parts of this paper?", "give me the table of contents",
  "what topics does this paper cover?", "walk me through the structure"

structural:
- asks for overview/summary/key takeaways/main points/contributions of the whole paper
- asks to explain or summarize a SPECIFIC SECTION or broad topic, e.g.:
  applications, use cases, implications, limitations, discussion, future work,
  conclusion, results, findings, methodology, approach, datasets, experiments
- asks for a brief description like "in 2–3 lines" about a broad part of the paper

references:
- asks for references/bibliography/citations/works cited/reference list

specific:
- asks for a single narrow fact/value/definition/equation/detail from a small passage

Output rules:
- Reply with ONLY one of these four words: outline, structural, references, specific
- No punctuation. No extra words. No explanation.
"""

_QA_SYSTEM_PROMPT = """\
You are a precise research assistant. You are given excerpts from a scientific \
paper and a question about it.

Rules:
- Answer ONLY from the provided excerpts. Do not use outside knowledge.
- The excerpts are tagged with a section name (e.g. "Abstract", "Introduction").
  Some excerpts may contain a small amount of text from an adjacent section due
  to shared page boundaries — ignore any content that is clearly not relevant
  to the question or the tagged section.
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


def _detect_section_filter(question: str, stored_headings: Optional[List[str]] = None) -> Optional[str]:
    """
    Detect if a question targets a specific named section and return the
    heading fragment to pass to search_chunks(section_heading=...).
    """
    q_lower = question.lower()

    if stored_headings:
        _strip_re = re.compile(r"^[IVXivx\d]+[.)]\s*|^[A-Za-z][.)]\s*")

        best_match: Optional[str] = None
        best_score = 0.0

        for heading in stored_headings:
            clean = _strip_re.sub("", heading).strip()
            words = [w for w in re.findall(r"[a-z]+", clean.lower()) if len(w) > 3]
            if not words:
                continue

            matched = sum(1 for w in words if w in q_lower)
            score = matched / len(words)

            if score > best_score and matched >= max(1, len(words) // 2):
                best_score = score
                best_match = clean

        if best_match and best_score >= 0.5:
            logger.debug(
                "[QA] Dynamic section filter: %r → %r (score=%.2f)",
                question[:60], best_match, best_score,
            )
            return best_match

    for question_kw, section_fragment in _GENERIC_SECTION_TRIGGER_MAP:
        if question_kw in q_lower:
            return section_fragment

    return None


def _is_reference_count_question(question: str) -> bool:
    q = (question or "").strip().lower()
    return (
        ("how many" in q and ("reference" in q or "citation" in q))
        or ("number of references" in q)
        or ("number of citations" in q)
        or ("total references" in q)
        or ("total citations" in q)
        or ("count the references" in q)
        or ("count of references" in q)
    )


def _is_last_reference_number_question(question: str) -> bool:
    q = (question or "").strip().lower()
    return (
        ("last reference number" in q)
        or ("highest reference number" in q)
        or ("largest reference number" in q)
        or ("final reference number" in q)
        or ("last citation number" in q)
        or ("highest citation number" in q)
        or ("largest citation number" in q)
    )


def _extract_requested_reference_number(question: str) -> Optional[int]:
    q = (question or "").strip().lower()

    patterns = [
        r"\b(?:reference|citation)\s*(?:number\s*)?(\d{1,4})\b",
        r"\b(?:show|give|find|get|fetch|list)\s+(?:me\s+)?(?:reference|citation)\s*(?:number\s*)?(\d{1,4})\b",
        r"\bwhat\s+is\s+(?:reference|citation)\s*(?:number\s*)?(\d{1,4})\b",
    ]

    for pattern in patterns:
        m = re.search(pattern, q)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                return None

    return None


class QAService:
    def __init__(self, store: PaperStore, llm_client) -> None:
        self._store = store
        self._llm = llm_client

    def _hard_route(self, question: str) -> QuestionType | None:
        """
        Cheap, deterministic bypass for references questions only.
        """
        q = (question or "").strip().lower()
        if not q:
            return None

        if any(k in q for k in _HARD_REFERENCES_KEYWORDS):
            return "references"

        return None

    def _classify_question(self, question: str) -> QuestionType:
        """
        Ask GPT-4o to classify the question as one of:
          'outline' | 'structural' | 'references' | 'specific'
        """
        hard = self._hard_route(question)
        if hard is not None:
            logger.info("[QA] Hard-routed question as '%s': %r", hard, question)
            return hard

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

            label = (raw or "").strip().lower().strip(" \t\r\n.:-\"'`")

            if label in ("outline", "structural", "references", "specific"):
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

        status = self._store.get_paper_status(arxiv_id)
        if status != STATUS_READY:
            if status == "processing":
                return "The paper is still being processed. Please try again in a moment."
            return "This paper hasn't been ingested yet. Open it first with `open <n>`."

        question_type = self._classify_question(question)

        if question_type == "outline":
            stored_json = self._store.get_headings(arxiv_id)
            if stored_json:
                tree = HeadingTree.from_json(stored_json)
                if not tree.is_empty():
                    logger.info(
                        "[QA] Outline fast-path: %d headings stored for %s",
                        len(tree.headings), arxiv_id,
                    )
                    display_outline = tree.format_for_display()
                    full_outline = tree.format_for_llm()

                    if not self._llm.enabled():
                        return display_outline

                    try:
                        answer = self._llm.chat(
                            messages=[
                                {"role": "system", "content": _HEADING_SYSTEM_PROMPT},
                                {
                                    "role": "user",
                                    "content": (
                                        (f"Paper: {paper_title}\n\n" if paper_title else "")
                                        + f"Heading outline:\n{full_outline}\n\n"
                                        + f"Question: {question}"
                                    ),
                                },
                            ],
                            temperature=0.0,
                        )
                        return answer.strip() if answer else display_outline
                    except Exception as e:
                        logger.error("[QA] Outline fast-path LLM call failed: %s", e)
                        return display_outline

            logger.info("[QA] Outline fast-path: no HeadingTree stored, falling back to RAG")
            question_type = "structural"

        # ── Fast path: answer simple references questions from stored metadata ──
        if question_type == "references":
            try:
                ref_meta = self._store.get_reference_metadata(arxiv_id)
            except Exception as e:
                logger.warning("[QA] get_reference_metadata failed for %s: %s", arxiv_id, e)
                ref_meta = None

            if ref_meta:
                ref_count = ref_meta.get("reference_count")
                last_ref_num = ref_meta.get("last_reference_number")
                ref_heading = ref_meta.get("reference_heading")
                ref_start_page = ref_meta.get("reference_start_page")
                ref_end_page = ref_meta.get("reference_end_page")

                if _is_reference_count_question(question):
                    if ref_count is not None:
                        if ref_heading and ref_start_page and ref_end_page:
                            if ref_start_page == ref_end_page:
                                return (
                                    f"There are {ref_count} references in the paper. "
                                    f"They appear under the '{ref_heading}' section on page {ref_start_page}."
                                )
                            return (
                                f"There are {ref_count} references in the paper. "
                                f"They appear under the '{ref_heading}' section on pages {ref_start_page}–{ref_end_page}."
                            )
                        return f"There are {ref_count} references in the paper."

                    if last_ref_num is not None:
                        return f"There are {last_ref_num} references in the paper."

                if _is_last_reference_number_question(question):
                    if last_ref_num is not None:
                        return f"The last reference number is {last_ref_num}."

                requested_ref_num = _extract_requested_reference_number(question)
                if requested_ref_num is not None and last_ref_num is not None:
                    if requested_ref_num > last_ref_num:
                        return (
                            f"Reference {requested_ref_num} does not exist in this paper. "
                            f"The last reference number is {last_ref_num}."
                        )

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

        section_filter = None
        stored_heading_texts: Optional[List[str]] = None
        if not is_references:
            try:
                headings_json = self._store.get_headings(arxiv_id)
                if headings_json:
                    heading_tree = HeadingTree.from_json(headings_json)
                    stored_heading_texts = [h.text for h in heading_tree.headings]
            except Exception:
                stored_heading_texts = None

            section_filter = _detect_section_filter(question, stored_heading_texts)
            if section_filter:
                top_k = None
                max_chars = SECTION_MAX_CONTEXT_CHARS
                system_prompt = _QA_SYSTEM_PROMPT
                logger.info(
                    "[QA] Section filter detected: %r → %r  (top_k=None, max_chars=%d)",
                    question, section_filter, max_chars,
                )

        logger.info(
            "[QA] arxiv_id=%s structural=%s references=%s section_filter=%r top_k=%s max_chars=%d question=%r",
            arxiv_id, is_structural, is_references, section_filter, top_k, max_chars, question,
        )

        try:
            query_vec = get_embedder().embed_query(question)
        except Exception as e:
            logger.error("[QA] embed_query failed: %s", e)
            return "Sorry, I had trouble processing your question. Please try again."

        try:
            hits = self._store.search_chunks(
                arxiv_id,
                query_vec,
                top_k=top_k,
                section_heading=section_filter,
            )
        except Exception as e:
            logger.error("[QA] search_chunks failed: %s", e)
            return "Sorry, I couldn't search the paper content. Please try again."

        if not hits:
            return "I have no content stored for this paper. Try re-opening it with `open <n>`."

        if is_references:
            tail_hits = []
            if hasattr(self._store, "get_tail_chunks"):
                try:
                    tail_hits = self._store.get_tail_chunks(arxiv_id, n=REF_TAIL_CHUNKS)
                except Exception as e:
                    logger.warning("[QA] get_tail_chunks failed: %s", e)
                    tail_hits = []

            if tail_hits:
                merged = {ci: (ci, text, score, sec) for ci, text, score, sec in hits}
                for ci, text, score, sec in tail_hits:
                    merged.setdefault(ci, (ci, text, score, sec))

                tail_indices = {ci for ci, _, _, _ in tail_hits}
                tail_part = sorted(
                    [v for v in merged.values() if v[0] in tail_indices],
                    key=lambda h: h[0],
                )
                other_part = sorted(
                    [v for v in merged.values() if v[0] not in tail_indices],
                    key=lambda h: -h[2],
                )
                hits = tail_part + other_part

        if is_structural or section_filter:
            hits = sorted(hits, key=lambda h: h[0])

        context_parts: List[str] = []
        total = 0
        for _ci, text, _score, _sec in hits:
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