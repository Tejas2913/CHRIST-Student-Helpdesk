import re
from typing import List, Dict, Any, Optional, Tuple

_CAMPUS_PATTERNS = {
    "Bangalore": [r"\bbangalore\b", r"\bbengaluru\b", r"\bkengeri\b", r"\bbannerghatta\b", r"\byeshwanthpur\b"],
    "Pune": [r"\bpune\b", r"\blavasa\b"],
    "Delhi NCR": [r"\bdelhi\b", r"\bncr\b", r"\bghaziabad\b"],
}
_PROGRAMME_PATTERNS = [
    (r"\b(post\s*graduate|postgraduate|\bpg\b|master'?s|m\.?tech|mba|mca|mcom|msc|ma|msw|llm|m\.?sc)\b", "PG"),
    (r"\b(under\s*graduate|undergraduate|\bug\b|bachelor'?s|b\.?tech|bba|bca|bcom|bsc|ba|llb|b\.?sc)\b", "UG"),
    (r"\b(ph\.?d|doctorate|doctoral)\b", "PhD"),
]

_FOLLOWUP_RE = re.compile(r"\b(it|that|this|those|these|the same|above|previous|second one|first one|third one|last one|earlier|mentioned)\b", re.I)
_COMPARE_RE = re.compile(r"\b(compare|difference|vs\.?|versus|between .+ and .+)\b", re.I)
_MULTI_RE = re.compile(r"\band\b|,|;|\?|(?:\n-)", re.I)


def _detect_campus(text: str) -> Optional[str]:
    t = text.lower()
    for campus, pats in _CAMPUS_PATTERNS.items():
        for p in pats:
            if re.search(p, t):
                return campus
    return None


def _detect_programme(text: str) -> Optional[str]:
    t = text.lower()
    for p, label in _PROGRAMME_PATTERNS:
        if re.search(p, t):
            return label
    # specific degree mentions can be used as-is for filtering hints in text search, but keep coarse label here
    m = re.search(r"\b(b\.?tech|m\.?tech|bba|mba|bcom|mcom|bca|mca|bsc|msc|ba|ma|llb|llm)\b", text, re.I)
    return m.group(1).upper().replace('.', '') if m else None


def _is_followup(message: str, history: Optional[List[Dict[str, str]]]) -> bool:
    if not history:
        return False
    msg = message.strip().lower()
    if len(msg.split()) <= 8 and _FOLLOWUP_RE.search(msg):
        return True
    if _FOLLOWUP_RE.search(msg) and len(msg.split()) <= 18:
        return True
    # short question likely referring to previous answer
    if msg.startswith(("what about", "how about", "and for", "and the", "what are its", "what is its")):
        return True
    return False


def _expand_query_terms(message: str) -> str:
    """Add lightweight domain synonyms/expansions to improve lexical and semantic retrieval."""
    q = (message or "").strip()
    low = q.lower()
    expansions = []
    if re.search(r"\b(pg|post\s*graduate|postgraduate|master'?s|m\.?sc|mba|mca|mtech|m\.?tech)\b", low):
        expansions.append("postgraduate PG")
    if re.search(r"\b(ug|under\s*graduate|undergraduate|bachelor'?s|b\.?sc|bba|bca|btech|b\.?tech)\b", low):
        expansions.append("undergraduate UG")
    if re.search(r"\b(borrow|issue|issued)\b", low) and re.search(r"\b(book|books)\b", low):
        expansions.extend(["library borrowing books issue", "loan period", "books allowed"])
    if re.search(r"\blate\b", low) and re.search(r"\b(return|returns|fine|penalty|overdue)\b", low):
        expansions.extend(["late return fine penalty overdue", "Rs per day"])
    if re.search(r"\battendance\b", low):
        expansions.extend(["minimum attendance requirement", "85%", "aggregate attendance"])
    if re.search(r"\b(leave of absence|medical leave|leave application|apply for leave)\b", low):
        expansions.extend(["leave application procedure", "student services", "absence"])
    if re.search(r"\bsemester\b", low) and re.search(r"\b(begin|start|commence|commencement)\b", low):
        expansions.extend(["commencement of classes", "semester begins", "academic calendar"])
    if re.search(r"\bhostel\b", low):
        expansions.append("hostel residence accommodation")
    if re.search(r"\bexam|examination\b", low):
        expansions.append("end semester examination ESE eligibility")
    # de-duplicate while preserving order
    seen = set()
    uniq = []
    for e in expansions:
        if e.lower() not in seen:
            uniq.append(e)
            seen.add(e.lower())
    if uniq:
        return q + "\nRelated terms: " + "; ".join(uniq)
    return q


def resolve_followup_query(message: str, history: Optional[List[Dict[str, Any]]]) -> str:
    """Create a retrieval-friendly query using recent conversation context for follow-ups."""
    msg = message.strip()
    if not history or not _is_followup(message, history):
        return _expand_query_terms(msg)
    # Use last user question and last assistant answer snippets as context anchors
    last_user = ""
    last_assistant = ""
    for m in reversed(history):
        role = (m.get("role") or "").lower()
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if role == "assistant" and not last_assistant:
            last_assistant = content
        elif role == "user" and not last_user:
            last_user = content
        if last_user and last_assistant:
            break
    # Try to extract a concrete referent from previous assistant answer (first heading/bullet/title-like line)
    referent = ""
    for line in (last_assistant or "").splitlines():
        sline = re.sub(r"^[\-\*\d\.\)\s]+", "", line).strip()
        sline = re.sub(r"\*\*|`", "", sline)
        if len(sline) >= 4:
            referent = sline[:120]
            break
    parts = [msg]
    if referent:
        parts.append(f"Context topic: {referent}")
    if last_user:
        parts.append(f"Previous question: {last_user[:180]}")
    combined = " | ".join(parts)
    return _expand_query_terms(combined)


def classify_query(message: str) -> str:
    q = message.lower().strip()
    if not q:
        return "factual"
    if re.search(r"\b(summarize|summary|overview|explain briefly|in short)\b", q):
        return "summarization"
    if _COMPARE_RE.search(q):
        return "comparison"
    # multi-part / multi-hop heuristics
    if q.count("?") >= 2 or len(re.findall(r"\b(and|also|as well as)\b", q)) >= 2 or re.search(r"\b(and|,).+\b(and|,).+\b", q):
        if any(k in q for k in ["how many", "when", "what is", "eligib", "penalt", "fee", "borrow", "period", "procedure", "criteria", "loan"]):
            return "multi_hop"
        return "multi_part"
    if re.search(r"\b(when|date|deadline|start|begin|end|calendar|semester begins|holiday)\b", q):
        return "date_calendar"
    if re.search(r"\b(eligib|criteria|qualify|requirement|required to|minimum|arrear|attendance)\b", q):
        return "eligibility"
    if re.search(r"\b(how many|how much|number of|percentage|%|marks?|fee|fine|penalty|borrow|limit|period|days|hours|cgpa|gpa|credits?|books?|loan)\b", q):
        if re.search(r"\b(book|books|borrow|loan|fine|penalty|fee|limit)\b", q):
            return "numerical"
        return "numerical"
    if re.search(r"\b(table|list of|schedule|breakdown)\b", q):
        return "table_lookup"
    if re.search(r"\b(policy|rule|regulation|code of conduct|misconduct|discipline|grievance|procedure|leave|exam|attendance)\b", q):
        if re.search(r"\b(how many|how much|percentage|%|marks?|fee|fine|penalty|borrow|limit|period|days|hours)\b", q):
            return "numerical"
        return "policy_regulation"
    if re.search(r"\b(campus|bangalore|bengaluru|kengeri|pune|lavasa|delhi ncr|ghaziabad)\b", q):
        return "campus_specific"
    return "factual"


def decompose_query(message: str, qtype: str) -> List[str]:
    q = message.strip()
    if qtype not in {"multi_hop", "multi_part", "comparison"}:
        return [q]
    # split on question marks and conjunction/comma patterns while keeping meaning
    parts = [p.strip(" ,;-") for p in re.split(r"\?+", q) if p.strip()]
    subqs: List[str] = []
    for p in parts:
        # further split "and"/commas if looks like multiple asks
        if qtype == "comparison" and _COMPARE_RE.search(p):
            subqs.append(p.strip() + ("?" if not p.endswith("?") else ""))
            continue
        chunks = re.split(r"\s*(?:,|;|\band also\b|\band\b|\bas well as\b)\s+", p, flags=re.I)
        # keep chunks that look like distinct asks
        cleaned = [c.strip() for c in chunks if len(c.strip()) >= 12]
        if len(cleaned) >= 2:
            # attach subject context from start if missing
            prefix = cleaned[0]
            for c in cleaned:
                sq = c
                if not re.search(r"\b(how|what|when|where|which|who|is|are|can|do|does)\b", c, re.I) and len(prefix.split()) <= 12:
                    sq = f"{prefix} {c}" if c not in prefix else c
                if not sq.endswith("?"):
                    sq += "?"
                subqs.append(sq.strip())
        else:
            if not p.endswith("?"):
                p += "?"
            subqs.append(p.strip())
    # dedupe preserving order
    out = []
    seen = set()
    for s in subqs:
        k = s.lower()
        if k not in seen:
            seen.add(k)
            out.append(s)
    return out[:5] if out else [q]


def analyze_query(message: str, history: Optional[List[Dict[str, Any]]] = None, request_filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    resolved = resolve_followup_query(message, history)
    qtype = classify_query(resolved)
    # refine type with followup context
    if _is_followup(message, history) and qtype == "factual":
        qtype = "follow_up"
    campus = _detect_campus(resolved) or (request_filters or {}).get("campus")
    programme = _detect_programme(resolved) or (request_filters or {}).get("programme")
    filters = {}
    for k in ["campus", "programme", "academic_year", "section", "content_type", "document", "document_version"]:
        v = (request_filters or {}).get(k)
        if v:
            filters[k] = v
    # do not force strict campus/programme metadata filters unless explicitly present in chunks likely; keep as hints
    hints = {"campus": campus, "programme": programme}

    # strategy hints
    strategies = ["vector", "bm25", "structured"]
    if qtype in {"numerical", "date_calendar", "table_lookup", "eligibility"}:
        strategies = ["structured", "bm25", "vector"]
    elif qtype in {"policy_regulation", "factual"}:
        strategies = ["vector", "bm25", "structured"]

    subqueries = decompose_query(resolved, qtype)
    use_multi_hop = qtype in {"multi_hop", "multi_part", "comparison"} or len(subqueries) > 1
    return {
        "original_query": message,
        "resolved_query": resolved,
        "query_type": qtype,
        "is_followup": _is_followup(message, history),
        "filters": filters,
        "hints": hints,
        "strategies": strategies,
        "subqueries": subqueries if use_multi_hop else [resolved],
        "use_multi_hop": use_multi_hop,
    }
