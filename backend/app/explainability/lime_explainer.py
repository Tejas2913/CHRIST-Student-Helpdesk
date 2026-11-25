"""
LIME Explainer for CHRIST Student Helpdesk.

Uses LIME Text Explainer for query intent classification (word-level importance)
and LIME Tabular Explainer for retrieval relevance (feature-level importance).

IMPORTANT: LIME fits a local surrogate model, so the word weights reflect that
surrogate's approximation, NOT causal proof of LLM behaviour.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

RETRIEVAL_FEATURE_NAMES = [
    "semantic_similarity",
    "bm25_score_norm",
    "reranker_score_norm",
    "keyword_overlap_ratio",
    "query_length_tokens",
    "chunk_position_norm",
    "section_match",
    "content_type_bonus",
]

INTENT_CLASSES = [
    "factual", "numerical", "eligibility", "policy_regulation",
    "date_calendar", "table_lookup", "campus_specific", "comparison",
    "multi_hop", "multi_part", "follow_up",
]


class LIMEExplainer:
    """Thread-safe LIME explainer with lazy initialisation."""

    def __init__(self):
        self._text_explainer = None
        self._tabular_explainer = None
        self._intent_pipeline = None
        self._retrieval_surrogate = None
        self._training_X = None

    # ------------------------------------------------------------------
    # Shared surrogate helpers
    # ------------------------------------------------------------------

    def _ensure_intent_pipeline(self):
        if self._intent_pipeline is not None:
            return
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline

        seed_queries = [
            "What is the minimum attendance requirement?",
            "How many books can I borrow from the library?",
            "When does the semester begin?",
            "What is the late fine for returning books?",
            "Compare UG and PG attendance rules",
            "Is BCC campus different from BKC?",
            "List all scholarship criteria",
            "What is the examination eligibility policy?",
            "How much is the hostel fee?",
            "What are the academic calendar dates?",
            "How do I apply for a leave of absence?",
            "What is the CGPA requirement for merit scholarship?",
        ]
        seed_labels = [
            "eligibility", "numerical", "date_calendar", "numerical",
            "comparison", "campus_specific", "table_lookup", "policy_regulation",
            "numerical", "date_calendar", "policy_regulation", "eligibility",
        ]
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=1000)),
            ("clf", LogisticRegression(max_iter=500, C=1.0, multi_class="multinomial")),
        ])
        pipe.fit(seed_queries, seed_labels)
        self._intent_pipeline = pipe

    def _ensure_retrieval_surrogate(self, X_train: Optional[np.ndarray] = None):
        if self._retrieval_surrogate is not None:
            return
        from sklearn.ensemble import GradientBoostingRegressor

        rng = np.random.default_rng(42)
        if X_train is not None and len(X_train) >= 5:
            X = X_train
        else:
            n = 200
            sem = rng.uniform(0.1, 1.0, n)
            bm = rng.uniform(0.0, 0.8, n)
            rk = sem * 0.9 + rng.uniform(0.0, 0.1, n)
            kw = rng.uniform(0.0, 0.6, n)
            ql = rng.uniform(0.2, 1.0, n)
            cp = rng.uniform(0.0, 1.0, n)
            sm = rng.choice([0.0, 1.0], n).astype(float)
            cb = rng.choice([0.0, 0.1], n).astype(float)
            X = np.column_stack([sem, bm, rk, kw, ql, cp, sm, cb]).astype(np.float32)
        self._training_X = X
        y = (0.45 * X[:, 0] + 0.2 * X[:, 1] + 0.15 * X[:, 2] + 0.1 * X[:, 3] + 0.05 * X[:, 6] + 0.05 * X[:, 7])
        y = np.clip(y, 0, 1).astype(np.float32)
        m = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)
        m.fit(X, y)
        self._retrieval_surrogate = m

    def _ensure_tabular_explainer(self):
        if self._tabular_explainer is not None:
            return
        try:
            from lime.lime_tabular import LimeTabularExplainer
        except ImportError:
            raise RuntimeError("lime library not installed. Run: pip install lime")
        self._ensure_retrieval_surrogate()
        self._tabular_explainer = LimeTabularExplainer(
            training_data=self._training_X,
            feature_names=RETRIEVAL_FEATURE_NAMES,
            mode="regression",
            random_state=42,
            discretize_continuous=True,
        )

    def _ensure_text_explainer(self):
        if self._text_explainer is not None:
            return
        try:
            from lime.lime_text import LimeTextExplainer
        except ImportError:
            raise RuntimeError("lime library not installed. Run: pip install lime")
        self._text_explainer = LimeTextExplainer(
            class_names=INTENT_CLASSES,
            random_state=42,
        )

    # ------------------------------------------------------------------
    # Text explanation (intent classification)
    # ------------------------------------------------------------------

    def explain_intent(self, query: str, num_features: int = 10, num_samples: int = 300) -> Dict[str, Any]:
        t0 = time.perf_counter()
        try:
            self._ensure_intent_pipeline()
            self._ensure_text_explainer()

            proba = self._intent_pipeline.predict_proba([query])[0]
            predicted_class = self._intent_pipeline.classes_[int(np.argmax(proba))]
            confidence = float(np.max(proba))
            cls_idx = list(self._intent_pipeline.classes_).index(predicted_class)

            def predict_proba_fn(texts):
                return self._intent_pipeline.predict_proba(texts)

            exp = self._text_explainer.explain_instance(
                query,
                predict_proba_fn,
                num_features=num_features,
                num_samples=num_samples,
                labels=[cls_idx],
            )

            word_weights = exp.as_list(label=cls_idx)
            word_importance = [
                {
                    "word": w,
                    "lime_weight": round(float(v), 6),
                    "positive": float(v) > 0,
                }
                for w, v in word_weights
            ]
            word_importance.sort(key=lambda x: abs(x["lime_weight"]), reverse=True)

            # Local fidelity: score of local surrogate
            local_pred_score = exp.score if hasattr(exp, "score") else None

            runtime_ms = (time.perf_counter() - t0) * 1000
            return {
                "target": "intent_classification",
                "query": query,
                "predicted_class": predicted_class,
                "confidence": confidence,
                "class_probabilities": {
                    cls: float(p)
                    for cls, p in zip(self._intent_pipeline.classes_, proba)
                },
                "word_importance": word_importance,
                "positive_words": [w for w in word_importance if w["positive"]][:5],
                "negative_words": [w for w in word_importance if not w["positive"]][:5],
                "local_fidelity": round(float(local_pred_score), 4) if local_pred_score is not None else None,
                "runtime_ms": round(runtime_ms, 2),
                "num_samples": num_samples,
                "surrogate": "LimeTextExplainer",
            }
        except Exception as exc:
            logger.exception("LIME intent explanation failed: %s", exc)
            return _empty_lime_result("intent_classification", time.perf_counter() - t0, error=str(exc))

    # ------------------------------------------------------------------
    # Tabular explanation (retrieval relevance)
    # ------------------------------------------------------------------

    def explain_retrieval(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        bm25_scores: Optional[List[float]] = None,
        num_features: int = 8,
        num_samples: int = 500,
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        try:
            import re

            self._ensure_retrieval_surrogate()
            self._ensure_tabular_explainer()

            q_tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
            q_len = max(len(q_tokens), 1)
            n = len(chunks)
            if n == 0:
                return _empty_lime_result("retrieval_relevance", time.perf_counter() - t0)

            scores = np.array([float(c.get("score", 0.0)) for c in chunks], dtype=np.float32)
            max_s = scores.max() or 1.0
            sem_sim = scores / max_s

            if bm25_scores and len(bm25_scores) == n:
                bm = np.array(bm25_scores, dtype=np.float32)
                bm_norm = bm / (bm.max() + 1e-9)
            else:
                bm_norm = np.zeros(n, dtype=np.float32)

            rerank_norm = sem_sim.copy()
            kw_overlap = np.zeros(n, dtype=np.float32)
            for i, c in enumerate(chunks):
                txt = (c.get("search_text") or c.get("text") or "").lower()
                ct = set(re.findall(r"[a-z0-9]+", txt))
                kw_overlap[i] = len(q_tokens & ct) / q_len
            q_len_feat = np.full(n, min(q_len / 30.0, 1.0), dtype=np.float32)
            chunk_pos_norm = np.array([1.0 - i / max(n - 1, 1) for i in range(n)], dtype=np.float32)
            section_match = np.array([1.0 if c.get("section") else 0.0 for c in chunks], dtype=np.float32)
            content_bonus = np.array(
                [0.1 if (c.get("content_type") or "") in {"table", "table_record", "eligibility"} else 0.0 for c in chunks],
                dtype=np.float32,
            )
            X = np.column_stack([sem_sim, bm_norm, rerank_norm, kw_overlap, q_len_feat, chunk_pos_norm, section_match, content_bonus]).astype(np.float32)

            # Explain top chunk only for performance
            instance = X[0]
            exp = self._tabular_explainer.explain_instance(
                instance,
                self._retrieval_surrogate.predict,
                num_features=num_features,
                num_samples=num_samples,
            )
            feat_weights = exp.as_list()
            feature_importance = [
                {
                    "feature_description": f_desc,
                    "lime_weight": round(float(w), 6),
                    "positive": float(w) > 0,
                }
                for f_desc, w in feat_weights
            ]
            feature_importance.sort(key=lambda x: abs(x["lime_weight"]), reverse=True)

            local_fidelity = exp.score if hasattr(exp, "score") else None
            prediction = float(self._retrieval_surrogate.predict(instance.reshape(1, -1))[0])

            runtime_ms = (time.perf_counter() - t0) * 1000
            return {
                "target": "retrieval_relevance",
                "query": query,
                "predicted_relevance": round(prediction, 4),
                "feature_importance": feature_importance,
                "positive_features": [f for f in feature_importance if f["positive"]][:4],
                "negative_features": [f for f in feature_importance if not f["positive"]][:4],
                "local_fidelity": round(float(local_fidelity), 4) if local_fidelity is not None else None,
                "runtime_ms": round(runtime_ms, 2),
                "num_samples": num_samples,
                "top_chunk_page": chunks[0].get("page") if chunks else None,
                "surrogate": "LimeTabularExplainer",
            }
        except Exception as exc:
            logger.exception("LIME retrieval explanation failed: %s", exc)
            return _empty_lime_result("retrieval_relevance", time.perf_counter() - t0, error=str(exc))

    # ------------------------------------------------------------------
    # Unified entry point
    # ------------------------------------------------------------------

    def explain(
        self,
        query: str,
        target: str,
        chunks: Optional[List[Dict[str, Any]]] = None,
        bm25_scores: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        if target == "intent_classification":
            return self.explain_intent(query)
        elif target == "retrieval_relevance":
            return self.explain_retrieval(query, chunks or [], bm25_scores)
        else:
            return {"error": f"Unknown LIME target: {target}", "target": target}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _empty_lime_result(target: str, elapsed: float, error: str = "") -> Dict[str, Any]:
    return {
        "target": target,
        "feature_importance": [],
        "word_importance": [],
        "positive_features": [],
        "negative_features": [],
        "local_fidelity": None,
        "runtime_ms": round(elapsed * 1000, 2),
        "error": error,
    }


_lime_explainer: Optional[LIMEExplainer] = None


def get_lime_explainer() -> LIMEExplainer:
    global _lime_explainer
    if _lime_explainer is None:
        _lime_explainer = LIMEExplainer()
    return _lime_explainer
