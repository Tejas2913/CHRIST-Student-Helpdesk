import os
import json
import pickle
import logging
import math
import re
from collections import Counter, defaultdict
from typing import List, Dict, Any, Optional, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.config import (
    PDF_PATH,
    FAISS_INDEX_PATH,
    CHUNKS_PATH,
    INDEX_META_PATH,
    STRUCTURED_RECORDS_PATH,
    EMBEDDING_MODEL_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K_RESULTS,
    RETRIEVAL_CANDIDATES,
    RERANK_TOP_N,
    INDEX_SCHEMA_VERSION,
    RERANKER_MODEL,
    settings,
)
from app.rag.extractor import extract_document, build_chunks

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[./-][A-Za-z0-9]+)?%?")


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _WORD_RE.findall(text or "")]


class _BM25Index:
    def __init__(self, docs_tokens: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs_tokens = docs_tokens
        self.doc_lens = [len(d) for d in docs_tokens]
        self.avgdl = sum(self.doc_lens) / len(self.doc_lens) if self.doc_lens else 0.0
        self.df: Dict[str, int] = defaultdict(int)
        self.postings: Dict[str, Dict[int, int]] = defaultdict(dict)
        for i, toks in enumerate(docs_tokens):
            counts = Counter(toks)
            for t, c in counts.items():
                self.df[t] += 1
                self.postings[t][i] = c
        self.N = len(docs_tokens)

    def scores(self, query: str) -> np.ndarray:
        q_tokens = _tokenize(query)
        if not q_tokens or self.N == 0:
            return np.zeros(self.N, dtype=np.float32)
        scores = np.zeros(self.N, dtype=np.float32)
        for term in q_tokens:
            if term not in self.postings:
                continue
            df = self.df[term]
            idf = math.log(1 + (self.N - df + 0.5) / (df + 0.5))
            posting = self.postings[term]
            for doc_idx, tf in posting.items():
                dl = self.doc_lens[doc_idx] or 1
                denom = tf + self.k1 * (1 - self.b + self.b * (dl / (self.avgdl or 1.0)))
                scores[doc_idx] += idf * (tf * (self.k1 + 1) / denom)
        return scores


_STOP_TOKENS_RERANK = {
    "what", "when", "where", "which", "who", "whom", "how", "many", "much", "the", "and", "for", "with", "from",
    "that", "this", "are", "can", "does", "do", "is", "am", "was", "were", "will", "would", "should", "could",
    "student", "students", "christ", "university", "please", "tell", "about", "there", "their", "have", "has",
    "need", "know", "regarding", "details", "information", "info", "related", "terms", "context", "topic", "previous",
    "question", "a", "an", "in", "on", "at", "to", "of", "or", "by", "as", "any", "be", "all", "its", "it",
}

_SYNONYMS_MAP = {
    "pg": {"pg", "postgraduate", "master", "masters"},
    "postgraduate": {"pg", "postgraduate", "master", "masters"},
    "ug": {"ug", "undergraduate", "bachelor", "bachelors"},
    "undergraduate": {"ug", "undergraduate", "bachelor", "bachelors"},
    "borrow": {"borrow", "issue", "issued", "loan", "lending", "allowed", "books"},
    "fine": {"fine", "penalty", "overdue", "charge"},
    "attendance": {"attendance", "present", "absent", "absence", "detention", "condonation", "85%"},
    "commence": {"commence", "commencement", "start", "begin", "inauguration", "reopening", "classes", "calendar"},
    "begin": {"commence", "commencement", "start", "begin", "inauguration", "reopening", "classes", "calendar"},
    "start": {"commence", "commencement", "start", "begin", "inauguration", "reopening", "classes", "calendar"},
    "exam": {"exam", "examination", "examinations", "ese", "cia", "assessment"},
    "examination": {"exam", "examination", "examinations", "ese", "cia", "assessment"},
    "examinations": {"exam", "examination", "examinations", "ese", "cia", "assessment"},
    "scholarship": {"scholarship", "fee concession", "merit", "financial aid"},
    "hostel": {"hostel", "residence", "accommodation"},
    "criteria": {"criteria", "eligibility", "requirements", "conditions", "requirement", "condition"},
    "eligibility": {"eligibility", "criteria", "requirements", "conditions", "requirement", "condition", "eligible"},
}


_CALENDAR_MONTHS = [
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december"
]


class _LightweightReranker:
    """Heuristic reranker combining RRF consensus rank with content overlap and domain bonuses."""

    def score(self, query: str, chunk: Dict[str, Any], base_score: float = 0.0) -> float:
        # Normalize RRF base score (typical RRF scores are ~0.015 - 0.048)
        base_norm = min(1.0, float(base_score) / 0.045) if base_score > 0 else 0.0

        all_toks = _tokenize(query)
        content_toks = [t for t in all_toks if t not in _STOP_TOKENS_RERANK and len(t) > 1]
        if not content_toks:
            content_toks = all_toks

        text = (chunk.get("search_text") or chunk.get("text") or "").lower()
        t_tokens = set(_tokenize(text))

        # Synonym-aware content token overlap
        matched_tokens = 0
        for tok in content_toks:
            syns = _SYNONYMS_MAP.get(tok, {tok})
            if syns & t_tokens or any(s in text for s in syns):
                matched_tokens += 1

        content_overlap = matched_tokens / max(len(content_toks), 1)

        # Bigram phrase bonus
        phrase_bonus = 0.0
        if len(content_toks) >= 2:
            for i in range(len(content_toks) - 1):
                bi = f"{content_toks[i]} {content_toks[i+1]}"
                if bi in text:
                    phrase_bonus = 0.15
                    break

        # Number / percentage alignment bonus
        num_bonus = 0.0
        q_nums = set(re.findall(r"\b\d+(?:\.\d+)?%?\b", query.lower()))
        if q_nums:
            t_nums = set(re.findall(r"\b\d+(?:\.\d+)?%?\b", text))
            if q_nums & t_nums:
                num_bonus = 0.20

        # Structured / Table / Intent bonus
        meta_bonus = 0.0
        ctype = (chunk.get("content_type") or "").lower()
        q_low = query.lower()
        if ctype in {"table", "table_record", "numeric_policy"} and any(
            k in q_low for k in ["how many", "number", "marks", "fee", "fine", "borrow", "limit", "period", "%", "percentage", "table"]
        ):
            meta_bonus += 0.12
        if ctype in {"date_event", "academic_calendar"} and any(
            k in q_low for k in ["when", "date", "calendar", "commence", "start", "begin", "schedule"]
        ):
            meta_bonus += 0.15
        if ctype in {"eligibility", "policy", "numeric_policy"} and any(
            k in q_low for k in ["eligibility", "criteria", "eligible", "requirement", "minimum", "rule", "policy"]
        ):
            meta_bonus += 0.12

        # Month and holiday calendar alignment
        month_bonus = 0.0
        target_months = [m for m in _CALENDAR_MONTHS if m in q_low]
        if target_months:
            t_month = target_months[0]
            has_target = t_month in text
            has_hol = "holiday" in text
            is_hol_q = any(k in q_low for k in ["holiday", "holidays", "vacation", "break", "day off"])
            other_m = [m for m in _CALENDAR_MONTHS if m != t_month and m in text]
            if has_target and is_hol_q and has_hol:
                month_bonus += 0.45
            elif has_target:
                month_bonus += 0.25
            elif other_m and not has_target:
                month_bonus -= 0.30

        # Final score balancing RRF consensus with content accuracy
        final_score = (base_norm * 0.45) + (content_overlap * 0.35) + phrase_bonus + num_bonus + meta_bonus + month_bonus
        return float(final_score)


class VectorStoreManager:
    _instance: Optional["VectorStoreManager"] = None

    def __init__(self):
        self.encoder: Optional[SentenceTransformer] = None
        self.index: Optional[faiss.Index] = None
        self.chunks: List[Dict[str, Any]] = []
        self.structured_records: List[Dict[str, Any]] = []
        self.bm25: Optional[_BM25Index] = None
        self._reranker = _LightweightReranker()
        self._cross_encoder = None

    @classmethod
    def get_instance(cls) -> "VectorStoreManager":
        if cls._instance is None:
            cls._instance = cls()
            cls._instance.initialize()
        return cls._instance

    def _init_encoder(self):
        if self.encoder is None:
            logger.info("Loading embedding model '%s'...", EMBEDDING_MODEL_NAME)
            os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
            os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
            self.encoder = SentenceTransformer(EMBEDDING_MODEL_NAME)
        # optional cross-encoder reranker
        if settings.enable_reranker and RERANKER_MODEL and self._cross_encoder is None:
            try:
                from sentence_transformers import CrossEncoder
                logger.info("Loading reranker model '%s'...", RERANKER_MODEL)
                self._cross_encoder = CrossEncoder(RERANKER_MODEL)
            except Exception as e:
                logger.warning("Failed to load reranker '%s', using lightweight reranker: %s", RERANKER_MODEL, e)
                self._cross_encoder = None

    def _index_meta(self) -> Dict[str, Any]:
        return {
            "schema_version": INDEX_SCHEMA_VERSION,
            "embedding_model": EMBEDDING_MODEL_NAME,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "pdf_path": str(PDF_PATH),
        }

    def _meta_matches(self) -> bool:
        try:
            if not os.path.exists(INDEX_META_PATH):
                return False
            with open(INDEX_META_PATH, "r", encoding="utf-8") as f:
                meta = json.load(f)
            return meta == self._index_meta()
        except Exception:
            return False

    def initialize(self, force_rebuild: bool = False):
        self._init_encoder()
        index_file = str(FAISS_INDEX_PATH)
        chunks_file = str(CHUNKS_PATH)
        records_file = str(STRUCTURED_RECORDS_PATH)

        if not force_rebuild and self._meta_matches() and os.path.exists(index_file) and os.path.exists(chunks_file):
            try:
                logger.info("Loading existing FAISS index from %s", index_file)
                self.index = faiss.read_index(index_file)
                with open(chunks_file, "rb") as f:
                    self.chunks = pickle.load(f)
                if os.path.exists(records_file):
                    with open(records_file, "r", encoding="utf-8") as f:
                        self.structured_records = json.load(f)
                else:
                    self.structured_records = [c for c in self.chunks if c.get("record")]
                self._build_bm25()
                logger.info("Loaded %s vectors, %s chunks, %s structured records", self.index.ntotal, len(self.chunks), len(self.structured_records))
                return
            except Exception as e:
                logger.warning("Could not load cached index (%s). Rebuilding...", e)

        self._build_index()

    def _build_index(self):
        logger.info("Building FAISS index from %s", PDF_PATH)
        extracted = extract_document(str(PDF_PATH))
        self.chunks = build_chunks(extracted, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
        self.structured_records = extracted.get("table_records", [])

        texts = [c.get("search_text") or c.get("text", "") for c in self.chunks]
        if not texts:
            raise RuntimeError("No chunks were generated from the handbook. Check PDF extraction.")

        embeddings = self.encoder.encode(texts, show_progress_bar=False, normalize_embeddings=True, batch_size=32)
        embeddings = np.array(embeddings).astype("float32")

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

        os.makedirs(os.path.dirname(str(FAISS_INDEX_PATH)), exist_ok=True)
        faiss.write_index(self.index, str(FAISS_INDEX_PATH))
        with open(CHUNKS_PATH, "wb") as f:
            pickle.dump(self.chunks, f)
        with open(STRUCTURED_RECORDS_PATH, "w", encoding="utf-8") as f:
            json.dump(self.structured_records, f, ensure_ascii=False)
        with open(INDEX_META_PATH, "w", encoding="utf-8") as f:
            json.dump(self._index_meta(), f, indent=2)

        self._build_bm25()
        logger.info("FAISS index built: %s vectors, %s chunks, %s tables, %s table records",
                    self.index.ntotal, len(self.chunks), len(extracted.get("tables", [])), len(self.structured_records))

    def _build_bm25(self):
        docs = [_tokenize((c.get("search_text") or c.get("text") or "")) for c in self.chunks]
        self.bm25 = _BM25Index(docs)

    @staticmethod
    def _passes_filters(chunk: Dict[str, Any], filters: Optional[Dict[str, Any]]) -> bool:
        if not filters:
            return True
        for key, val in filters.items():
            if val is None or val == "":
                continue
            cval = chunk.get(key)
            if cval is None:
                return False
            if isinstance(val, (list, tuple, set)):
                allowed = {str(v).lower() for v in val}
                if str(cval).lower() not in allowed:
                    return False
            else:
                if str(cval).lower() != str(val).lower():
                    return False
        return True

    def _vector_search(self, query: str, top_n: int, filters: Optional[Dict[str, Any]]) -> List[Tuple[int, float]]:
        if self.index is None or not self.chunks:
            self.initialize()
        # overfetch then filter
        k = min(max(top_n * 4, top_n), len(self.chunks))
        qv = self.encoder.encode([query], normalize_embeddings=True)
        qv = np.array(qv).astype("float32")
        scores, idxs = self.index.search(qv, k)
        out = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            if not self._passes_filters(self.chunks[idx], filters):
                continue
            out.append((int(idx), float(score)))
            if len(out) >= top_n:
                break
        return out

    def _bm25_search(self, query: str, top_n: int, filters: Optional[Dict[str, Any]]) -> List[Tuple[int, float]]:
        if self.bm25 is None:
            self._build_bm25()
        scores = self.bm25.scores(query)
        if scores.size == 0:
            return []
        ranked = np.argsort(-scores)
        out = []
        for idx in ranked:
            s = float(scores[idx])
            if s <= 0:
                continue
            idx = int(idx)
            if not self._passes_filters(self.chunks[idx], filters):
                continue
            out.append((idx, s))
            if len(out) >= top_n:
                break
        return out

    def _structured_search(self, query: str, top_n: int, filters: Optional[Dict[str, Any]]) -> List[Tuple[int, float]]:
        q = query.lower()
        q_tokens = set(_tokenize(query))
        q_nums = set(re.findall(r"\b\d+(?:\.\d+)?%?\b", q))
        scored = []
        for idx, c in enumerate(self.chunks):
            ctype = (c.get("content_type") or "").lower()
            if ctype not in {"table", "table_record", "eligibility", "numeric_policy", "date_event", "academic_calendar"}:
                continue
            if not self._passes_filters(c, filters):
                continue
            text = (c.get("search_text") or c.get("text") or "").lower()
            t_tokens = set(_tokenize(text))
            overlap = len(q_tokens & t_tokens)
            if overlap == 0 and not q_nums:
                continue
            score = overlap / max(len(q_tokens), 1)
            t_nums = set(re.findall(r"\b\d+(?:\.\d+)?%?\b", text))
            if q_nums and (q_nums & t_nums):
                score += 0.8
            if ctype in {"table", "table_record"}:
                score += 0.15
            if any(k in q for k in ["date", "when", "semester", "calendar"]) and ctype in {"date_event", "academic_calendar"}:
                score += 0.2
            if any(k in q for k in ["eligibility", "criteria", "eligible"]) and ctype == "eligibility":
                score += 0.2
            if score > 0:
                scored.append((idx, float(score)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_n]

    @staticmethod
    def _rrf_fuse(rank_lists: List[List[Tuple[int, float]]], k: int = 60) -> Dict[int, float]:
        fused: Dict[int, float] = defaultdict(float)
        for lst in rank_lists:
            for rank, (idx, _score) in enumerate(lst, start=1):
                fused[idx] += 1.0 / (k + rank)
        return fused

    def _rerank(self, query: str, candidate_idxs: List[int], base_scores: Dict[int, float]) -> List[Tuple[int, float]]:
        if not candidate_idxs:
            return []
        if self._cross_encoder is not None:
            try:
                pairs = [(query, (self.chunks[i].get("search_text") or self.chunks[i].get("text") or "")[:1200]) for i in candidate_idxs]
                ce_scores = self._cross_encoder.predict(pairs)
                ranked = sorted(((idx, float(s)) for idx, s in zip(candidate_idxs, ce_scores)), key=lambda x: x[1], reverse=True)
                return ranked
            except Exception as e:
                logger.warning("Cross-encoder rerank failed, falling back to lightweight reranker: %s", e)
        scored = []
        for idx in candidate_idxs:
            s = self._reranker.score(query, self.chunks[idx], base_scores.get(idx, 0.0))
            scored.append((idx, s))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def hybrid_search(
        self,
        query: str,
        top_k: int = TOP_K_RESULTS,
        filters: Optional[Dict[str, Any]] = None,
        candidate_k: Optional[int] = None,
        rerank_top_n: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if self.index is None or not self.chunks:
            self.initialize()
        candidate_k = candidate_k or max(settings.retrieval_candidates, top_k * 4)
        rerank_top_n = rerank_top_n or max(settings.rerank_top_n, top_k)

        vec = self._vector_search(query, candidate_k, filters)
        bm25 = self._bm25_search(query, candidate_k, filters)
        struct = self._structured_search(query, candidate_k, filters)

        fused = self._rrf_fuse([vec, bm25, struct])
        # fallback to vector-only if nothing fused
        if not fused and vec:
            fused = {idx: 1.0 / (60 + i) for i, (idx, _) in enumerate(vec, start=1)}

        # keep a reasonably broad candidate set for reranking
        prelim = sorted(fused.items(), key=lambda x: x[1], reverse=True)[: max(candidate_k, rerank_top_n * 3)]
        candidate_idxs = [idx for idx, _ in prelim]
        reranked = self._rerank(query, candidate_idxs, fused)

        results = []
        seen_keys = set()
        for idx, score in reranked:
            chunk = self.chunks[idx].copy()
            key = (chunk.get("page"), (chunk.get("text") or "")[:120])
            if key in seen_keys:
                continue
            seen_keys.add(key)
            chunk["score"] = float(score)
            # back-compat fields expected by existing callers
            chunk.setdefault("chunk_id", idx)
            results.append(chunk)
            if len(results) >= top_k:
                break

        # if reranker over-pruned (e.g. duplicates), fill from fused order
        if len(results) < top_k:
            for idx, score in prelim:
                if any(r is self.chunks[idx] or (r.get("page") == self.chunks[idx].get("page") and r.get("text") == self.chunks[idx].get("text")) for r in results):
                    continue
                chunk = self.chunks[idx].copy()
                chunk["score"] = float(score)
                chunk.setdefault("chunk_id", idx)
                results.append(chunk)
                if len(results) >= top_k:
                    break
        return results

    def search(self, query: str, top_k: int = TOP_K_RESULTS, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Backward-compatible search entrypoint; now hybrid by default."""
        return self.hybrid_search(query=query, top_k=top_k, filters=filters)
