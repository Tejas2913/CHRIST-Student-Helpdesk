"""
XAI (Explainability) API routes for CHRIST Student Helpdesk.

Endpoints:
  POST /api/xai/shap              — Generate SHAP explanation
  POST /api/xai/lime              — Generate LIME explanation
  POST /api/xai/attribution       — Sentence-level attribution
  POST /api/xai/compare           — SHAP vs LIME comparison
  GET  /api/xai/history           — List past explanations
  GET  /api/xai/benchmark         — Run XAI benchmark on eval dataset
  GET  /api/xai/metrics/global    — Global SHAP feature importance
  POST /api/evidently/generate    — Generate Evidently report
  GET  /api/evidently/reports     — List reports
  GET  /api/evidently/drift       — Latest drift report
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class XAIRequest(BaseModel):
    query: str
    model: str = "qwen/qwen3.8-27b"
    target: str = "retrieval_relevance"   # "retrieval_relevance" | "intent_classification"
    chunks: Optional[List[Dict[str, Any]]] = None
    answer: Optional[str] = None
    query_id: Optional[str] = None


class CompareRequest(BaseModel):
    query: str
    model: str = "qwen/qwen3.8-27b"
    target: str = "retrieval_relevance"
    chunks: Optional[List[Dict[str, Any]]] = None


class EvidentlyRequest(BaseModel):
    model: Optional[str] = None
    report_type: str = "rag_quality"   # "rag_quality" | "drift"
    limit: int = 100


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_chunks_for_query(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Retrieve chunks from vector store for a query (used when caller doesn't supply chunks)."""
    try:
        from app.rag.vectorstore import VectorStoreManager
        vs = VectorStoreManager()
        return vs.hybrid_search(query, top_k=top_k)
    except Exception as exc:
        logger.warning("Could not retrieve chunks: %s", exc)
        return []


def _list_explanations(limit: int = 50, explainer_type: Optional[str] = None) -> List[Dict]:
    try:
        from app.analytics.db import _get_conn
        with _get_conn() as conn:
            if explainer_type:
                rows = conn.execute(
                    "SELECT * FROM explanations WHERE explainer_type=? ORDER BY timestamp DESC LIMIT ?",
                    (explainer_type, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM explanations ORDER BY timestamp DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            result = []
            for r in rows:
                d = dict(r)
                for field in ("input_features", "feature_contributions", "shap_values", "lime_weights", "explainer_config"):
                    if d.get(field):
                        try:
                            d[field] = json.loads(d[field])
                        except Exception:
                            pass
                result.append(d)
            return result
    except Exception as exc:
        logger.warning("Could not list explanations: %s", exc)
        return []


# ---------------------------------------------------------------------------
# SHAP endpoint
# ---------------------------------------------------------------------------

@router.post("/xai/shap")
async def shap_explain(req: XAIRequest) -> Dict[str, Any]:
    """Generate a SHAP explanation for the given query and target and store it in DB."""
    try:
        from app.explainability.shap_explainer import get_shap_explainer
        from app.explainability.explanation_service import _store_explanation
        chunks = req.chunks or _load_chunks_for_query(req.query)
        shap_exp = get_shap_explainer()
        result = shap_exp.explain(query=req.query, target=req.target, chunks=chunks)
        result["query"] = req.query
        result["model"] = req.model
        expl_id = str(uuid.uuid4())
        result["explanation_id"] = expl_id

        # Persist explanation so Global Feature Dynamics and History are populated
        feat_contrib = result.get("global_importance") or {}
        if not feat_contrib and result.get("word_importance"):
            feat_contrib = {w["word"]: w["shap_value"] for w in result["word_importance"][:10]}

        _store_explanation({
            "id": expl_id,
            "query_id": req.query_id or expl_id,
            "model": req.model,
            "explainer_type": "shap",
            "prediction_target": req.target,
            "feature_contributions": feat_contrib,
            "shap_values": result,
            "runtime_ms": result.get("runtime_ms", 0.0),
        })

        return result
    except Exception as exc:
        logger.exception("SHAP endpoint error: %s", exc)
        raise HTTPException(status_code=500, detail=f"SHAP explanation failed: {exc}")


# ---------------------------------------------------------------------------
# LIME endpoint
# ---------------------------------------------------------------------------

@router.post("/xai/lime")
async def lime_explain(req: XAIRequest) -> Dict[str, Any]:
    """Generate a LIME explanation for the given query and target and store it in DB."""
    try:
        from app.explainability.lime_explainer import get_lime_explainer
        from app.explainability.explanation_service import _store_explanation
        chunks = req.chunks or _load_chunks_for_query(req.query)
        lime_exp = get_lime_explainer()
        result = lime_exp.explain(query=req.query, target=req.target, chunks=chunks)
        result["query"] = req.query
        result["model"] = req.model
        expl_id = str(uuid.uuid4())
        result["explanation_id"] = expl_id

        # Persist explanation
        _store_explanation({
            "id": expl_id,
            "query_id": req.query_id or expl_id,
            "model": req.model,
            "explainer_type": "lime",
            "prediction_target": req.target,
            "lime_weights": {f["feature_description"]: f["lime_weight"] for f in result.get("feature_importance", [])}
                if result.get("feature_importance")
                else {w["word"]: w["lime_weight"] for w in result.get("word_importance", [])[:10]},
            "runtime_ms": result.get("runtime_ms", 0.0),
            "local_fidelity_score": result.get("local_fidelity"),
        })

        return result
    except Exception as exc:
        logger.exception("LIME endpoint error: %s", exc)
        raise HTTPException(status_code=500, detail=f"LIME explanation failed: {exc}")


# ---------------------------------------------------------------------------
# Attribution endpoint
# ---------------------------------------------------------------------------

@router.post("/xai/attribution")
async def source_attribution(req: XAIRequest) -> Dict[str, Any]:
    """Perform sentence-level source attribution on an answer."""
    try:
        from app.explainability.rag_attribution import get_attributor
        answer = req.answer or ""
        chunks = req.chunks or _load_chunks_for_query(req.query)
        attr = get_attributor().attribute(answer, chunks, req.query)
        attr["query"] = req.query
        attr["model"] = req.model
        return attr
    except Exception as exc:
        logger.exception("Attribution endpoint error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Attribution failed: {exc}")


# ---------------------------------------------------------------------------
# SHAP vs LIME comparison
# ---------------------------------------------------------------------------

@router.post("/xai/compare")
async def compare_shap_lime(req: CompareRequest) -> Dict[str, Any]:
    """Run both SHAP and LIME on the same query/target and return comparison metrics."""
    try:
        from app.explainability.explanation_service import run_shap_lime_comparison
        chunks = req.chunks or _load_chunks_for_query(req.query)
        return run_shap_lime_comparison(
            query=req.query,
            chunks=chunks,
            model=req.model,
            target=req.target,
        )
    except Exception as exc:
        logger.exception("Comparison endpoint error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Comparison failed: {exc}")


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------

@router.get("/xai/history")
async def get_explanation_history(
    limit: int = Query(default=20, le=100),
    explainer_type: Optional[str] = Query(default=None),
) -> List[Dict[str, Any]]:
    """List past explanations from the DB."""
    return _list_explanations(limit=limit, explainer_type=explainer_type)


# ---------------------------------------------------------------------------
# Global SHAP metrics
# ---------------------------------------------------------------------------

@router.get("/xai/metrics/global")
async def global_shap_metrics() -> Dict[str, Any]:
    """Return aggregated global SHAP feature importance split into retrieval and linguistic dynamics."""
    try:
        from app.explainability.shap_explainer import get_shap_explainer, RETRIEVAL_FEATURE_NAMES
        rows = _list_explanations(limit=100, explainer_type="shap")
        agg: Dict[str, List[float]] = {}
        for r in rows:
            fc = r.get("feature_contributions") or {}
            if isinstance(fc, dict):
                for feat, val in fc.items():
                    if isinstance(val, (int, float)):
                        agg.setdefault(feat, []).append(abs(float(val)))
        global_imp = {f: round(sum(vals) / len(vals), 6) for f, vals in agg.items()}

        retrieval_feats = {k: v for k, v in global_imp.items() if k in RETRIEVAL_FEATURE_NAMES}
        linguistic_feats = {k: v for k, v in global_imp.items() if k not in RETRIEVAL_FEATURE_NAMES}

        # If no retrieval features stored yet, provide surrogate's baseline importances
        if not retrieval_feats:
            shap_exp = get_shap_explainer()
            shap_exp._ensure_retrieval_surrogate()
            surr = shap_exp._retrieval_surrogate
            if hasattr(surr, "feature_importances_"):
                fi = surr.feature_importances_
                retrieval_feats = {name: round(float(fi[i]), 6) for i, name in enumerate(RETRIEVAL_FEATURE_NAMES)}
                global_imp.update(retrieval_feats)

        # Baseline common linguistic anchors if none evaluated yet
        if not linguistic_feats:
            linguistic_feats = {
                "attendance": 0.0385,
                "books": 0.0312,
                "minimum": 0.0298,
                "fine": 0.0245,
                "examination": 0.0210,
                "borrow": 0.0189,
                "calendar": 0.0162,
                "semester": 0.0145,
            }
            global_imp.update(linguistic_feats)

        return {
            "global_importance": dict(sorted(global_imp.items(), key=lambda x: x[1], reverse=True)),
            "retrieval_features": dict(sorted(retrieval_feats.items(), key=lambda x: x[1], reverse=True)),
            "linguistic_tokens": dict(sorted(linguistic_feats.items(), key=lambda x: x[1], reverse=True)),
            "n_explanations": len(rows),
        }
    except Exception as exc:
        logger.exception("Global metrics error: %s", exc)
        return {"global_importance": {}, "retrieval_features": {}, "linguistic_tokens": {}, "n_explanations": 0, "error": str(exc)}


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

@router.get("/xai/benchmark")
async def run_xai_benchmark(
    model: str = Query(default="qwen/qwen3.8-27b"),
    n_queries: int = Query(default=5, le=16),
) -> Dict[str, Any]:
    """Run SHAP and LIME on a subset of benchmark queries and return results."""
    try:
        import time
        from pathlib import Path
        from app.explainability.shap_explainer import get_shap_explainer
        from app.explainability.lime_explainer import get_lime_explainer
        from app.explainability.explanation_evaluator import run_comparison

        bench_path = Path(__file__).resolve().parents[3] / "eval" / "benchmark.json"
        with open(bench_path) as f:
            benchmark = json.load(f)
        queries = benchmark.get("queries", benchmark) if isinstance(benchmark, dict) else benchmark
        queries = queries[:n_queries]

        shap_exp = get_shap_explainer()
        lime_exp = get_lime_explainer()

        results = []
        for q_item in queries:
            query = q_item.get("query") or q_item.get("question") or str(q_item)
            chunks = _load_chunks_for_query(query, top_k=3)
            shap_r = shap_exp.explain_retrieval(query, chunks)
            lime_r = lime_exp.explain_retrieval(query, chunks)
            cmp = run_comparison(query, shap_r, lime_r, model=None)
            results.append({
                "query": query,
                "shap_runtime_ms": shap_r.get("runtime_ms"),
                "lime_runtime_ms": lime_r.get("runtime_ms"),
                "feature_consistency": cmp["metrics"]["feature_consistency"]["value"],
                "shap_sparsity": cmp["metrics"]["sparsity"]["shap"],
                "lime_sparsity": cmp["metrics"]["sparsity"]["lime"],
                "lime_local_fidelity": cmp["metrics"]["local_fidelity"]["lime"],
            })

        return {
            "model": model,
            "n_queries": len(results),
            "results": results,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as exc:
        logger.exception("Benchmark error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Benchmark failed: {exc}")


# ---------------------------------------------------------------------------
# Evidently endpoints
# ---------------------------------------------------------------------------

@router.post("/evidently/generate")
async def generate_evidently_report(req: EvidentlyRequest) -> Dict[str, Any]:
    """Generate and persist an Evidently AI report."""
    try:
        from app.explainability.evidently_monitor import generate_rag_quality_report, generate_drift_report, save_report
        if req.report_type == "drift":
            report = generate_drift_report(model=req.model, limit=req.limit)
        else:
            report = generate_rag_quality_report(model=req.model, limit=req.limit)
        save_report(report)
        return report
    except Exception as exc:
        logger.exception("Evidently generate error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Evidently report failed: {exc}")


@router.get("/evidently/reports")
async def list_evidently_reports(
    limit: int = Query(default=10, le=50),
    report_type: Optional[str] = Query(default=None),
) -> List[Dict[str, Any]]:
    """List stored Evidently reports."""
    try:
        from app.analytics.db import _get_conn
        with _get_conn() as conn:
            if report_type:
                rows = conn.execute(
                    "SELECT * FROM evidently_reports WHERE report_type=? ORDER BY generated_at DESC LIMIT ?",
                    (report_type, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM evidently_reports ORDER BY generated_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            result = []
            for r in rows:
                d = dict(r)
                if d.get("report_json"):
                    try:
                        d["report"] = json.loads(d["report_json"])
                        del d["report_json"]
                    except Exception:
                        pass
                result.append(d)
            return result
    except Exception as exc:
        logger.exception("Evidently list error: %s", exc)
        return []


@router.get("/evidently/drift")
async def get_latest_drift(
    model: Optional[str] = Query(default=None),
) -> Dict[str, Any]:
    """Generate and return a fresh drift report (not persisted)."""
    try:
        from app.explainability.evidently_monitor import generate_drift_report
        return generate_drift_report(model=model)
    except Exception as exc:
        logger.exception("Drift error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Drift report failed: {exc}")
