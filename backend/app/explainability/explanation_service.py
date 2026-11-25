"""
Explanation Service — orchestrates SHAP, LIME, attribution, and evaluation.
Acts as the single entry-point for XAI operations and stores results in the DB.
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _store_explanation(record: Dict[str, Any]) -> None:
    try:
        from app.analytics.db import get_connection
        with get_connection() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO explanations
                   (id, query_id, model, explainer_type, prediction_target,
                    input_features, feature_contributions, shap_values, lime_weights,
                    runtime_ms, faithfulness_score, stability_score, sparsity_score,
                    local_fidelity_score, timestamp, model_version, explainer_config)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    record.get("id", str(uuid.uuid4())),
                    record.get("query_id"),
                    record.get("model"),
                    record.get("explainer_type"),
                    record.get("prediction_target"),
                    json.dumps(record.get("input_features", {})),
                    json.dumps(record.get("feature_contributions", {})),
                    json.dumps(record.get("shap_values", {})),
                    json.dumps(record.get("lime_weights", {})),
                    record.get("runtime_ms"),
                    record.get("faithfulness_score"),
                    record.get("stability_score"),
                    record.get("sparsity_score"),
                    record.get("local_fidelity_score"),
                    record.get("timestamp", datetime.now(timezone.utc).isoformat()),
                    record.get("model_version"),
                    json.dumps(record.get("explainer_config", {})),
                ),
            )
            conn.commit()
    except Exception as exc:
        logger.warning("Could not store explanation: %s", exc)


def _store_attribution(record: Dict[str, Any]) -> None:
    try:
        from app.analytics.db import get_connection
        for sa in record.get("sentence_attributions", []):
            with get_connection() as conn:
                conn.execute(
                    """INSERT OR REPLACE INTO attributions
                       (id, query_id, model, sentence, attributed_chunk_id,
                        attributed_page, attributed_section, supporting_passage,
                        attribution_confidence, is_supported, timestamp)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        str(uuid.uuid4()),
                        record.get("query_id"),
                        record.get("model"),
                        sa.get("sentence", ""),
                        str(sa.get("attributed_chunk_id", "")),
                        sa.get("attributed_page"),
                        sa.get("attributed_section", ""),
                        sa.get("supporting_passage", ""),
                        sa.get("similarity_score", 0.0),
                        1 if sa.get("support_level") == "supported" else 0,
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )
                conn.commit()
    except Exception as exc:
        logger.warning("Could not store attribution: %s", exc)


def run_full_explanation(
    query: str,
    answer: str,
    chunks: List[Dict[str, Any]],
    model: str = "unknown",
    query_id: Optional[str] = None,
    targets: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Run all enabled explainability targets for a single query.
    Returns a comprehensive explanation dict.
    Fails gracefully — never raises.
    """
    if targets is None:
        targets = ["retrieval_relevance", "intent_classification", "attribution"]

    result: Dict[str, Any] = {
        "query_id": query_id or str(uuid.uuid4()),
        "query": query,
        "model": model,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # 1. SHAP
    if "retrieval_relevance" in targets or "intent_classification" in targets:
        try:
            from app.explainability.shap_explainer import get_shap_explainer
            shap_exp = get_shap_explainer()

            if "retrieval_relevance" in targets and chunks:
                shap_ret = shap_exp.explain_retrieval(query, chunks)
                result["shap_retrieval"] = shap_ret
                _store_explanation({
                    "id": str(uuid.uuid4()),
                    "query_id": result["query_id"],
                    "model": model,
                    "explainer_type": "shap",
                    "prediction_target": "retrieval_relevance",
                    "feature_contributions": shap_ret.get("global_importance", {}),
                    "shap_values": shap_ret,
                    "runtime_ms": shap_ret.get("runtime_ms"),
                })

            if "intent_classification" in targets:
                shap_int = shap_exp.explain_intent(query)
                result["shap_intent"] = shap_int
                _store_explanation({
                    "id": str(uuid.uuid4()),
                    "query_id": result["query_id"],
                    "model": model,
                    "explainer_type": "shap",
                    "prediction_target": "intent_classification",
                    "feature_contributions": {"predicted_class": shap_int.get("predicted_class")},
                    "shap_values": shap_int,
                    "runtime_ms": shap_int.get("runtime_ms"),
                })
        except Exception as exc:
            logger.warning("SHAP explanation skipped: %s", exc)
            result["shap_error"] = str(exc)

    # 2. LIME
    try:
        from app.explainability.lime_explainer import get_lime_explainer
        lime_exp = get_lime_explainer()

        if "intent_classification" in targets:
            lime_int = lime_exp.explain_intent(query)
            result["lime_intent"] = lime_int

        if "retrieval_relevance" in targets and chunks:
            lime_ret = lime_exp.explain_retrieval(query, chunks)
            result["lime_retrieval"] = lime_ret
            _store_explanation({
                "id": str(uuid.uuid4()),
                "query_id": result["query_id"],
                "model": model,
                "explainer_type": "lime",
                "prediction_target": "retrieval_relevance",
                "lime_weights": {f["feature_description"]: f["lime_weight"] for f in lime_ret.get("feature_importance", [])},
                "runtime_ms": lime_ret.get("runtime_ms"),
                "local_fidelity_score": lime_ret.get("local_fidelity"),
            })
    except Exception as exc:
        logger.warning("LIME explanation skipped: %s", exc)
        result["lime_error"] = str(exc)

    # 3. Attribution
    if "attribution" in targets and answer and chunks:
        try:
            from app.explainability.rag_attribution import get_attributor
            attr = get_attributor().attribute(answer, chunks, query)
            result["attribution"] = attr
            _store_attribution({
                "query_id": result["query_id"],
                "model": model,
                **attr,
            })
        except Exception as exc:
            logger.warning("Attribution skipped: %s", exc)
            result["attribution_error"] = str(exc)

    return result


def run_shap_lime_comparison(
    query: str,
    chunks: List[Dict[str, Any]],
    model: str = "unknown",
    target: str = "retrieval_relevance",
) -> Dict[str, Any]:
    """Run both SHAP and LIME on the same target and produce a comparison report."""
    try:
        from app.explainability.shap_explainer import (
            get_shap_explainer,
            RETRIEVAL_FEATURE_NAMES,
            _extract_retrieval_features,
        )
        from app.explainability.lime_explainer import get_lime_explainer
        from app.explainability.explanation_evaluator import run_comparison

        shap_exp = get_shap_explainer()
        lime_exp = get_lime_explainer()

        surrogate_model = None
        instance = None
        feature_names = None

        if target == "retrieval_relevance":
            shap_result = shap_exp.explain_retrieval(query, chunks)
            lime_result = lime_exp.explain_retrieval(query, chunks)
            shap_exp._ensure_retrieval_surrogate()
            surrogate_model = shap_exp._retrieval_surrogate
            if chunks:
                X = _extract_retrieval_features(query, chunks)
                instance = X[0]
            feature_names = RETRIEVAL_FEATURE_NAMES
        else:
            shap_result = shap_exp.explain_intent(query)
            lime_result = lime_exp.explain_intent(query)
            shap_exp._ensure_intent_surrogate()
            surrogate_model = shap_exp._intent_surrogate

        comparison = run_comparison(
            query,
            shap_result,
            lime_result,
            model=surrogate_model,
            instance=instance,
            feature_names=feature_names,
        )
        comparison["shap_detail"] = shap_result
        comparison["lime_detail"] = lime_result
        return comparison
    except Exception as exc:
        logger.exception("SHAP vs LIME comparison failed: %s", exc)
        return {"error": str(exc), "query": query}
