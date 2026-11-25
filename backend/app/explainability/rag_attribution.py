"""
RAG Source Attribution — sentence-level attribution of answer statements to retrieved chunks.

For each sentence in the generated answer:
  1. Encode it with the same embedding model used by the vector store.
  2. Compute cosine similarity with each retrieved chunk.
  3. If the best-matching chunk exceeds the threshold, mark the sentence as SUPPORTED
     and return the page, section, and supporting passage.
  4. If no chunk exceeds the threshold, mark the sentence as UNSUPPORTED.

IMPORTANT: Similarity ≠ proof of factual correctness. This module only identifies
structural overlap between the answer text and retrieved evidence.
"""
from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

SUPPORT_THRESHOLD = 0.30  # cosine similarity to consider a sentence "supported"
PARTIAL_THRESHOLD = 0.18  # partial support
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences; keep non-trivial sentences (≥ 5 words)."""
    raw = _SENT_SPLIT_RE.split(text.strip())
    out = []
    for s in raw:
        s = s.strip()
        if s and len(s.split()) >= 5:
            out.append(s)
    return out or [text.strip()]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class RAGAttributor:
    """Lazy-loaded embedding-based sentence attributor."""

    def __init__(self):
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                from app.core.config import EMBEDDING_MODEL_NAME
                self._model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            except Exception as exc:
                logger.error("Could not load embedding model for attribution: %s", exc)
                raise
        return self._model

    def attribute(
        self,
        answer: str,
        chunks: List[Dict[str, Any]],
        query: str = "",
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        try:
            model = self._get_model()
            sentences = _split_sentences(answer)

            if not chunks or not sentences:
                return _empty_attribution(answer, time.perf_counter() - t0)

            # Encode all chunks at once for efficiency
            chunk_texts = [
                (c.get("search_text") or c.get("text") or "")[:512] for c in chunks
            ]
            chunk_embs = model.encode(chunk_texts, normalize_embeddings=True, show_progress_bar=False)
            sent_embs = model.encode(sentences, normalize_embeddings=True, show_progress_bar=False)

            sentence_attributions = []
            total_supported = 0
            total_partial = 0

            for i, (sent, s_emb) in enumerate(zip(sentences, sent_embs)):
                # Find best matching chunk
                sims = [_cosine(s_emb, c_emb) for c_emb in chunk_embs]
                best_idx = int(np.argmax(sims))
                best_sim = sims[best_idx]
                best_chunk = chunks[best_idx]

                if best_sim >= SUPPORT_THRESHOLD:
                    support_level = "supported"
                    total_supported += 1
                elif best_sim >= PARTIAL_THRESHOLD:
                    support_level = "partial"
                    total_partial += 1
                else:
                    support_level = "unsupported"

                chunk_text = (best_chunk.get("search_text") or best_chunk.get("text") or "")
                # Extract the most relevant passage (first 300 chars)
                supporting_passage = chunk_text[:300].strip() if support_level != "unsupported" else ""

                sentence_attributions.append(
                    {
                        "sentence_id": i,
                        "sentence": sent,
                        "support_level": support_level,
                        "similarity_score": round(best_sim, 4),
                        "attributed_chunk_id": best_chunk.get("chunk_id", best_idx),
                        "attributed_page": best_chunk.get("page"),
                        "attributed_section": best_chunk.get("section", ""),
                        "attributed_subsection": best_chunk.get("subsection", ""),
                        "document": best_chunk.get("document", ""),
                        "supporting_passage": supporting_passage,
                        "all_chunk_sims": [round(s, 3) for s in sims],
                    }
                )

            n = len(sentences)
            coverage = {
                "total_sentences": n,
                "supported": total_supported,
                "partial": total_partial,
                "unsupported": n - total_supported - total_partial,
                "support_rate": round(total_supported / n, 3) if n else 0.0,
                "coverage_rate": round((total_supported + total_partial) / n, 3) if n else 0.0,
            }

            runtime_ms = (time.perf_counter() - t0) * 1000
            return {
                "query": query,
                "answer_preview": answer[:200],
                "sentence_attributions": sentence_attributions,
                "coverage": coverage,
                "runtime_ms": round(runtime_ms, 2),
                "support_threshold": SUPPORT_THRESHOLD,
                "partial_threshold": PARTIAL_THRESHOLD,
                "n_chunks": len(chunks),
            }
        except Exception as exc:
            logger.exception("RAG attribution failed: %s", exc)
            return _empty_attribution(answer, time.perf_counter() - t0, error=str(exc))


def _empty_attribution(answer: str, elapsed: float, error: str = "") -> Dict[str, Any]:
    return {
        "query": "",
        "answer_preview": answer[:200],
        "sentence_attributions": [],
        "coverage": {
            "total_sentences": 0, "supported": 0, "partial": 0,
            "unsupported": 0, "support_rate": 0.0, "coverage_rate": 0.0,
        },
        "runtime_ms": round(elapsed * 1000, 2),
        "support_threshold": SUPPORT_THRESHOLD,
        "partial_threshold": PARTIAL_THRESHOLD,
        "n_chunks": 0,
        "error": error,
    }


_attributor: Optional[RAGAttributor] = None


def get_attributor() -> RAGAttributor:
    global _attributor
    if _attributor is None:
        _attributor = RAGAttributor()
    return _attributor
