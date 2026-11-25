import time
import re
import logging
from typing import List, Dict, Any, Optional, Tuple

from app.core.config import SYSTEM_PROMPT, settings
from app.rag.vectorstore import VectorStoreManager
from app.rag.query_analysis import analyze_query
from app.llm.factory import LLMProviderFactory
from app.monitoring.logger import LLMOpsLogger
from app.schemas.chat import (
    ChatRequest,
    ChatResponse,
    SourceCitation,
    LatencyMetrics,
    RetrievalDebug,
)

logger = logging.getLogger(__name__)

_UNANSWERABLE_PHRASE = "I couldn't find this information in the available university handbook/document."
_PARTIAL_PHRASE = "I couldn't verify this part in the available university handbook/document."

_STOP_TOKENS = {
    "what", "when", "where", "which", "who", "whom", "how", "many", "much", "the", "and", "for", "with", "from",
    "that", "this", "are", "can", "does", "do", "is", "am", "was", "were", "will", "would", "should", "could",
    "student", "students", "christ", "university", "please", "tell", "about", "there", "their", "have", "has",
    "need", "know", "regarding", "details", "information", "info", "related", "terms", "context", "topic", "previous",
    "question",
}


def _safe_int_id(val: Any) -> int:
    if isinstance(val, int):
        return val
    try:
        return int(val)
    except (ValueError, TypeError):
        return abs(hash(str(val))) % 1000000


def _source_key(s: Dict[str, Any]) -> Tuple[Any, str]:
    txt = (s.get("text") or "").strip()
    return (s.get("page"), re.sub(r"\s+", " ", txt[:180]).lower())


def _dedupe_sources(sources: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for s in sorted(sources, key=lambda x: float(x.get("score", 0.0)), reverse=True):
        k = _source_key(s)
        if k in seen:
            continue
        seen.add(k)
        out.append(s)
        if len(out) >= limit:
            break
    return out


def _format_evidence(sources: List[Dict[str, Any]]) -> str:
    blocks = []
    for i, s in enumerate(sources, 1):
        label_parts = [f"Evidence {i}", f"Page {s.get('page')}"]
        if s.get("document"):
            label_parts.append(str(s.get("document")))
        if s.get("section"):
            label_parts.append(str(s.get("section")))
        if s.get("subsection"):
            label_parts.append(str(s.get("subsection")))
        if s.get("content_type"):
            label_parts.append(f"type={s.get('content_type')}")
        if s.get("table_name"):
            label_parts.append(f"table={s.get('table_name')}")
        header = " | ".join(label_parts)
        cite = f"[Handbook p. {s.get('page')}]"
        text = (s.get("text") or "").strip()
        blocks.append(f"{header}\nCitation label: {cite}\n{text}")
    return "\n\n---\n\n".join(blocks)


def _query_tokens(query: str) -> List[str]:
    toks = [t for t in re.findall(r"[a-z0-9]+", (query or "").lower()) if len(t) > 2]
    key = [t for t in toks if t not in _STOP_TOKENS]
    return key or toks


def _evidence_relevance(query: str, sources: List[Dict[str, Any]]) -> Tuple[float, float, float]:
    if not sources:
        return 0.0, 0.0, 0.0
    scores = [float(s.get("score", 0.0)) for s in sources]
    top = scores[0]
    avg = sum(scores[:3]) / min(len(scores), 3)
    qset = set(_query_tokens(query))
    combined = " ".join((s.get("search_text") or s.get("text") or "") for s in sources[:3]).lower()
    overlap = sum(1 for t in qset if t in combined)
    overlap_ratio = overlap / max(len(qset), 1)
    return top, avg, overlap_ratio


def _filter_relevant_sources(query: str, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop obviously irrelevant retrieved chunks before confidence/prompting."""
    if not sources:
        return []
    q = (query or "").lower()
    qset = set(_query_tokens(query))
    q_nums = set(re.findall(r"\b\d+(?:\.\d+)?%?\b", q))
    filtered: List[Dict[str, Any]] = []
    for s in sources:
        text = ((s.get("search_text") or s.get("text") or "")).lower()
        tset = set(re.findall(r"[a-z0-9]+", text))
        overlap = len(qset & tset) / max(len(qset), 1)
        score = float(s.get("score", 0.0))
        t_nums = set(re.findall(r"\b\d+(?:\.\d+)?%?\b", text))
        num_align = bool(q_nums and (q_nums & t_nums))

        keep = False
        if score >= 0.45 and overlap >= 0.10:
            keep = True
        elif score >= 0.28 and overlap >= 0.18:
            keep = True
        elif overlap >= 0.30:
            keep = True
        elif score >= 0.22 and overlap >= 0.22:
            keep = True
        elif num_align and score >= 0.18 and overlap >= 0.08:
            keep = True
        elif score >= 0.70:
            keep = True

        if keep:
            filtered.append(s)

    if not filtered and sources:
        top, _avg, overlap_ratio = _evidence_relevance(query, sources)
        # keep best plausible evidence for table/policy lookups
        if (top >= 0.20 and overlap_ratio >= 0.20) or (top >= 0.32 and overlap_ratio >= 0.10) or top >= 0.45:
            filtered = sources[:2]
    return filtered


def _estimate_confidence(query: str, sources: List[Dict[str, Any]], qtype: str) -> Tuple[str, str]:
    if not sources:
        return "unanswerable", "No relevant evidence retrieved."

    top, avg, overlap_ratio = _evidence_relevance(query, sources)
    combined = " ".join((s.get("search_text") or s.get("text") or "") for s in sources[:3]).lower()

    numeric_or_date_query = bool(re.search(
        r"\b(how many|how much|percentage|%|\d|date|when|deadline|begin|start|marks?|fee|fine|borrow|limit|period|days|hours|credit|credits|cgpa|gpa|late|penalt)\b",
        (query or "").lower(),
    ))
    has_numeric_evidence = bool(re.search(r"\b\d+(?:\.\d+)?%?\b", combined))

    # clearly irrelevant retrieval -> unanswerable (do not guess)
    if top < 0.18 and overlap_ratio < 0.10:
        return "unanswerable", "Retrieved evidence is not sufficiently relevant to the question."
    if overlap_ratio < 0.06 and top < 0.40:
        return "unanswerable", "Retrieved evidence does not meaningfully match the question."

    strong = (
        (top >= 0.75 and overlap_ratio >= 0.18)
        or (top >= 0.95 and overlap_ratio >= 0.12)
        or (avg >= 0.55 and overlap_ratio >= 0.20)
        or (overlap_ratio >= 0.40 and top >= 0.20)
        or (overlap_ratio >= 0.30 and has_numeric_evidence and top >= 0.18)
    )
    medium = (
        (top >= 0.38 and overlap_ratio >= 0.10)
        or (avg >= 0.28 and overlap_ratio >= 0.12)
        or (top >= 0.50 and overlap_ratio >= 0.08)
        or (overlap_ratio >= 0.22 and top >= 0.18)
        or (numeric_or_date_query and has_numeric_evidence and overlap_ratio >= 0.18)
    )

    if numeric_or_date_query and not has_numeric_evidence:
        if strong:
            return "partial", "Relevant passages found but explicit numeric/date evidence is weak."
        if medium:
            return "partial", "Retrieved evidence may be related but explicit numeric/date support is limited."
        return "unanswerable", "Numeric/date question without clear supporting value in retrieved evidence."

    if strong:
        return "high", "Top evidence strongly matches the query."
    if medium:
        return "partial", "Some relevant evidence found, but support may be incomplete."
    return "low", "Retrieved evidence is weak or only loosely related."


def _build_user_prompt(
    original_query: str,
    resolved_query: str,
    evidence_text: str,
    history: Optional[List[Dict[str, Any]]],
    analysis: Dict[str, Any],
    confidence: str,
    confidence_reason: str,
) -> str:
    hist_lines = []
    if history:
        recent = history[-6:]
        for m in recent:
            role = (m.get("role") or "").strip().lower()
            content = (m.get("content") or "").strip()
            if not content:
                continue
            if role not in {"user", "assistant"}:
                continue
            content = re.sub(r"\s+", " ", content)
            if len(content) > 400:
                content = content[:400] + "..."
            hist_lines.append(f"{role.title()}: {content}")
    history_block = "\n".join(hist_lines) if hist_lines else "(none)"

    subqs = analysis.get("subqueries") or [original_query]
    subq_block = "\n".join(f"- {q}" for q in subqs)
    guidance = {
        "high": "Provide a direct, complete answer with citations for factual statements.",
        "partial": (
            "Answer only the parts supported by evidence and explicitly state what could not be verified. "
            f"Use this sentence for missing parts when appropriate: \"{_PARTIAL_PHRASE}\""
        ),
        "low": (
            "Do not guess. If evidence is insufficient, respond clearly that the information is not found: "
            f"\"{_UNANSWERABLE_PHRASE}\""
        ),
        "unanswerable": (
            f"Respond clearly: \"{_UNANSWERABLE_PHRASE}\" "
            "Advise contacting the relevant university office if appropriate."
        ),
    }.get(confidence, "Answer using evidence only.")

    return (
        f"Conversation history (for interpreting follow-ups only; do not treat as evidence):\n{history_block}\n\n"
        f"User question:\n{original_query}\n\n"
        f"Resolved retrieval query:\n{resolved_query}\n\n"
        f"Detected query type: {analysis.get('query_type')}\n"
        f"Sub-questions to cover (if applicable):\n{subq_block}\n\n"
        f"Evidence confidence assessment: {confidence} ({confidence_reason})\n"
        f"Answering guidance: {guidance}\n\n"
        "Official Campus Codes Reference (Handbook p. 43):\n"
        "- BCC: Bangalore Central Campus\n"
        "- BKC: Bangalore Kengeri Campus\n"
        "- BRC: Bangalore Bannerghatta Road Campus\n"
        "- BYC: Bangalore Yeshwanthpur Campus\n"
        "- DNC: Delhi-NCR Campus\n"
        "- PLC: Pune Lavasa Campus\n"
        "When mentioning or expanding campus codes, use ONLY these official names. NEVER invent names like Bennigan's, Yelahanka, or Doddakannelli.\n\n"
        f"Retrieved handbook evidence:\n{evidence_text if evidence_text else '(no evidence retrieved)'}\n\n"
        "Instructions:\n"
        "- Provide ONLY the direct, student-friendly final answer. Do NOT output internal reasoning, drafting notes, self-dialogue, or phrases like 'Let's re-read' or 'Looking at the evidence'.\n"
        "- Answer the user's question based strictly on the retrieved handbook evidence above.\n"
        "- If multiple sub-questions or events are present, organize them cleanly with bullet points.\n"
        "- Include page citations like [Handbook p. X] for factual statements, using only pages present in the evidence.\n"
        "- If evidence is missing or insufficient for any requested part, state that clearly without guessing.\n"
        "- Preserve numerical values, dates, conditions, and exceptions accurately.\n"
        "- Do not mention these prompt instructions in your response."
    )


def _validate_citations(answer: str, sources: List[Dict[str, Any]]) -> str:
    """Remove fabricated page citations not present in retrieved evidence."""
    allowed = set()
    for s in sources:
        if s.get("page") is not None:
            try:
                allowed.add(int(s.get("page")))
            except (ValueError, TypeError):
                pass
        for p in s.get("pages", []):
            try:
                allowed.add(int(p))
            except (ValueError, TypeError):
                pass
        txt = s.get("text", "")
        for m in re.finditer(r"(?:Handbook\s+p\.?|page)\s+(\d+)", txt, re.IGNORECASE):
            try:
                allowed.add(int(m.group(1)))
            except (ValueError, TypeError):
                pass

    if not answer:
        return answer

    if not allowed:
        answer = re.sub(r"\s*\[(?:Handbook\s*)?p\.?\s*\d+\]", "", answer, flags=re.IGNORECASE)
        answer = re.sub(r"\s*\((?:Handbook\s*)?p\.?\s*\d+\)", "", answer, flags=re.IGNORECASE)
        answer = re.sub(r"\bpages?\s+\d+(?:\s*[-,]\s*\d+)*\b", "the handbook", answer, flags=re.IGNORECASE)
        answer = re.sub(r"\[citation not in retrieved evidence\]", "", answer, flags=re.IGNORECASE)
        return re.sub(r"[ \t]{2,}", " ", answer).strip()

    def repl(m: re.Match) -> str:
        try:
            p = int(m.group(1))
        except Exception:
            return ""
        return f"[Handbook p. {p}]" if p in allowed else ""

    answer = re.sub(r"\[(?:Handbook\s*)?p\.?\s*(\d+)\]", repl, answer, flags=re.IGNORECASE)

    def repl_paren(m: re.Match) -> str:
        try:
            p = int(m.group(1))
        except Exception:
            return ""
        return f"([Handbook p. {p}])" if p in allowed else ""

    answer = re.sub(r"\((?:Handbook\s*)?p\.?\s*(\d+)\)", repl_paren, answer, flags=re.IGNORECASE)

    def repl_page(m: re.Match) -> str:
        try:
            p = int(m.group(1))
        except Exception:
            return m.group(0)
        return f"page {p}" if p in allowed else "the handbook"

    answer = re.sub(r"\bpage\s+(\d+)\b", repl_page, answer, flags=re.IGNORECASE)
    answer = re.sub(r"\[citation not in retrieved evidence\]", "", answer, flags=re.IGNORECASE)
    answer = re.sub(r"\s+([,.;:])", r"\1", answer)
    answer = re.sub(r"\(\s*\)", "", answer)
    answer = re.sub(r"\[\s*\]", "", answer)
    answer = re.sub(r"\n{3,}", "\n\n", answer)
    answer = re.sub(r"[ \t]{2,}", " ", answer)
    return answer.strip()


def _clean_reasoning_traces(answer: str) -> str:
    """Strip accidental reasoning traces or self-dialogue if an instruction-following model slips into reasoning."""
    if not answer:
        return answer
    lines = answer.split("\n")
    cleaned = []
    skip = False
    for line in lines:
        l_low = line.strip().lower()
        if any(l_low.startswith(p) for p in [
            "let's re-read", "let's check", "let's inspect", "let me re-read", "looking at evidence",
            "wait, looking at", "evidence 2 shows", "evidence 3 shows", "let's verify", "re-reading"
        ]):
            skip = True
            continue
        if skip and (
            line.strip().startswith(("#", "**", "-", "*", "1.", "2.", "3.", "•"))
            or any(l_low.startswith(p) for p in ["therefore", "based on", "in summary", "according to", "the holidays"])
        ):
            skip = False
        if not skip:
            cleaned.append(line)

    result = "\n".join(cleaned).strip()
    return result if result else answer


def _correct_campus_abbreviations(answer: str) -> str:
    """Ensure campus abbreviations expand to official CHRIST University names and clean up hallucinations."""
    if not answer:
        return answer
    replacements = [
        (r"\bBCC\s*\([^)]*Bennigan[^)]*\)", "BCC (Bangalore Central Campus)"),
        (r"\bBYC\s*\([^)]*Bennigan[^)]*\)", "BYC (Bangalore Yeshwanthpur Campus)"),
        (r"\bBYC\s*\([^)]*Yelahanka[^)]*\)", "BYC (Bangalore Yeshwanthpur Campus)"),
        (r"\bDNC\s*\([^)]*Doddakannelli[^)]*\)", "DNC (Delhi-NCR Campus)"),
        (r"\bBKC\s*\([^)]*Bengaluru Campus[^)]*\)", "BKC (Bangalore Kengeri Campus)"),
        (r"\bBCC\b(?!\s*\()", "BCC (Bangalore Central Campus)"),
        (r"\bBYC\b(?!\s*\()", "BYC (Bangalore Yeshwanthpur Campus)"),
        (r"\bDNC\b(?!\s*\()", "DNC (Delhi-NCR Campus)"),
        (r"\bBKC\b(?!\s*\()", "BKC (Bangalore Kengeri Campus)"),
        (r"\bBRC\b(?!\s*\()", "BRC (Bangalore Bannerghatta Road Campus)"),
        (r"\bPLC\b(?!\s*\()", "PLC (Pune Lavasa Campus)"),
    ]
    for pattern, repl in replacements:
        answer = re.sub(pattern, repl, answer)
    return answer


class RAGPipeline:
    def __init__(self):
        self.vector_store = VectorStoreManager.get_instance()
        self.provider_factory = LLMProviderFactory.get_instance()

    def _retrieve(self, query: str, analysis: Dict[str, Any], top_k: int) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
        filters = analysis.get("filters") or {}
        strategies = list(analysis.get("strategies") or ["vector", "bm25", "structured"])
        used_strategies = list(strategies)
        subqueries = analysis.get("subqueries") or [query]

        all_sources: List[Dict[str, Any]] = []
        if analysis.get("use_multi_hop") and settings.enable_multi_hop and len(subqueries) > 1:
            used_strategies.append("multi_hop")
            per_q_k = max(3, min(top_k, 4))
            for sq in subqueries:
                sq_sources = self.vector_store.hybrid_search(sq, top_k=per_q_k, filters=filters)
                for s in sq_sources:
                    s.setdefault("_matched_subquery", sq)
                all_sources.extend(sq_sources)
            # combined retrieval pass
            all_sources.extend(self.vector_store.hybrid_search(query, top_k=top_k, filters=filters))
        else:
            all_sources.extend(self.vector_store.hybrid_search(query, top_k=max(top_k, 5), filters=filters))

        merged = _dedupe_sources(all_sources, limit=max(top_k, 6))
        return merged, subqueries, used_strategies

    def process_chat(self, request: ChatRequest) -> ChatResponse:
        t_start = time.perf_counter()

        query = (request.message or "").strip()
        history = [
            m.model_dump() if hasattr(m, "model_dump") else (m.dict() if hasattr(m, "dict") else dict(m))
            for m in (request.conversation_history or [])
        ]
        top_k = request.top_k or settings.top_k_results
        req_provider = request.provider or self.provider_factory.active_provider_name
        req_model = request.model or self.provider_factory.active_model_name

        request_filters = {
            "campus": request.campus,
            "programme": request.programme,
            "academic_year": request.academic_year,
            "section": request.section,
            "content_type": request.content_type,
        }
        request_filters = {k: v for k, v in request_filters.items() if v}

        # 1. Query analysis / follow-up resolution
        analysis = analyze_query(query, history=history, request_filters=request_filters)
        resolved_query = analysis.get("resolved_query") or query

        # 2. Retrieval (hybrid + optional multi-hop)
        t_ret_start = time.perf_counter()
        raw_sources, subqueries, strategies = self._retrieve(resolved_query, analysis, top_k=top_k)
        # use base query (without expansion hints) for relevance scoring
        scoring_query = (
            resolved_query.split("\nRelated terms:")[0]
            .split("\n[Retrieval hints]:")[0]
            .strip() or query
        )
        filtered_sources = _filter_relevant_sources(scoring_query, raw_sources)
        if filtered_sources:
            raw_sources = filtered_sources
        retrieval_ms = (time.perf_counter() - t_ret_start) * 1000

        # 3. Confidence estimation
        confidence, confidence_reason = _estimate_confidence(
            scoring_query, raw_sources, analysis.get("query_type", "factual")
        )
        # treat very weak evidence as unanswerable to reduce hallucination risk
        if confidence == "low":
            top_s, _avg_s, ov_s = _evidence_relevance(scoring_query, raw_sources)
            if top_s < 0.30 and ov_s < 0.12:
                confidence, confidence_reason = "unanswerable", confidence_reason + " Evidence is too weak to answer reliably."
            else:
                confidence = "partial"

        # 4. Build grounded prompt
        evidence_text = _format_evidence(raw_sources) if raw_sources else ""
        user_content = _build_user_prompt(
            original_query=query,
            resolved_query=resolved_query,
            evidence_text=evidence_text,
            history=history,
            analysis=analysis,
            confidence=confidence,
            confidence_reason=confidence_reason,
        )

        source_citations = [
            SourceCitation(
                chunk_id=_safe_int_id(s.get("numeric_id") or s.get("chunk_id", 0)),
                id=str(s.get("id") or s.get("chunk_id") or ""),
                page=int(s.get("page") or 0),
                score=round(float(s.get("score", 0.0)), 4),
                text=s.get("text", ""),
                document=s.get("document"),
                document_version=s.get("document_version"),
                academic_year=s.get("academic_year"),
                section=s.get("section"),
                subsection=s.get("subsection"),
                clause=s.get("clause"),
                content_type=s.get("content_type"),
                table_name=s.get("table_name"),
                campus=s.get("campus"),
                programme=s.get("programme"),
            )
            for s in raw_sources
        ]

        # 5. Generation
        t_gen_start = time.perf_counter()
        provider = self.provider_factory.get_provider(req_provider)

        answer = ""
        model_used = req_model
        token_count = None
        error_msg = None
        status = "SUCCESS"

        try:
            if confidence == "unanswerable" or not raw_sources:
                answer = _UNANSWERABLE_PHRASE + " Please contact the relevant CHRIST University office for confirmation."
                model_used = req_model
                token_count = 0
                generation_ms = (time.perf_counter() - t_gen_start) * 1000
            else:
                answer, model_used, token_count = provider.generate(
                    prompt=user_content,
                    system_prompt=SYSTEM_PROMPT,
                    model=req_model,
                    max_tokens=1500,
                )
                generation_ms = (time.perf_counter() - t_gen_start) * 1000
                answer = _validate_citations(answer, raw_sources)
                answer = _clean_reasoning_traces(answer)
                answer = _correct_campus_abbreviations(answer)
                if not (answer or "").strip():
                    answer = _UNANSWERABLE_PHRASE + " Please contact the relevant CHRIST University office for confirmation."
                    confidence = "unanswerable"
                    confidence_reason = "Generation returned empty output."
        except Exception as e:
            status = "ERROR"
            error_msg = str(e)
            logger.error("Generation error using %s/%s: %s", req_provider, req_model, e)
            answer = (
                f"**Inference Error ({req_provider.upper()})**\n\n"
                f"{str(e)}\n\n"
                "You can switch between Groq (Cloud) and Ollama (Local) in the model settings."
            )
            generation_ms = (time.perf_counter() - t_gen_start) * 1000

        total_ms = (time.perf_counter() - t_start) * 1000

        # 6. Logging
        try:
            # Build enriched fields for analytics
            _source_pages = [s.get("page") for s in raw_sources if s.get("page") is not None]
            _chunk_ids = [str(s.get("chunk_id", "")) for s in raw_sources]
            _answer_preview = (answer or "")[:200]
            LLMOpsLogger.log_request(
                query=query,
                provider=req_provider,
                model=model_used,
                retrieval_latency_ms=retrieval_ms,
                generation_latency_ms=generation_ms,
                total_latency_ms=total_ms,
                sources_retrieved=len(source_citations),
                status=status,
                token_count=token_count,
                error=error_msg,
                query_type=analysis.get("query_type"),
                answer_preview=_answer_preview,
                confidence=confidence,
                source_pages=_source_pages,
                retrieved_chunk_ids=_chunk_ids,
            )
        except Exception:
            logger.exception("Failed to log request")

        display_sources = [] if confidence == "unanswerable" else source_citations

        debug = RetrievalDebug(
            query_type=analysis.get("query_type"),
            confidence=confidence,
            confidence_reason=confidence_reason,
            resolved_query=resolved_query,
            subqueries=subqueries,
            filters=analysis.get("filters") or {},
            retrieval_strategy=strategies,
            evidence_count=len(source_citations),
        )

        return ChatResponse(
            answer=answer,
            sources=display_sources,
            provider=req_provider,
            model=model_used,
            latency=LatencyMetrics(
                retrieval_ms=round(retrieval_ms, 2),
                generation_ms=round(generation_ms, 2),
                total_ms=round(total_ms, 2),
            ),
            error=error_msg,
            confidence=confidence,
            query_type=analysis.get("query_type"),
            debug=debug,
        )
