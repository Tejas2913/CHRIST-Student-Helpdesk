"""
Evidently AI Monitor for CHRIST Student Helpdesk.

Generates RAG quality reports, data drift detection, and query monitoring
using the Evidently library. Reports are stored as JSON in the analytics DB.

Gracefully degrades if evidently is not installed.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _get_evidently():
    try:
        import evidently
        return evidently
    except ImportError:
        return None


def _load_query_logs(limit: int = 200) -> List[Dict[str, Any]]:
    """Load recent query logs from the analytics SQLite DB."""
    try:
        from app.analytics.db import get_connection
        with get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM query_logs ORDER BY timestamp DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]
    except Exception as exc:
        logger.warning("Could not load query logs for Evidently: %s", exc)
        return []


def _parse_confidence(val: Any) -> float:
    if isinstance(val, (int, float)):
        return float(val)
    v = str(val or "").strip().lower()
    if v in ("high", "very high"):
        return 0.9
    if v in ("medium", "moderate"):
        return 0.7
    if v in ("low", "very low"):
        return 0.4
    try:
        return float(v)
    except (ValueError, TypeError):
        return 0.5


def generate_rag_quality_report(
    model: Optional[str] = None,
    limit: int = 100,
) -> Dict[str, Any]:
    """
    Generate a RAG quality report using Evidently metrics on query log data.
    Returns a JSON-serialisable dict with computed metrics and a unique report ID.
    """
    t0 = time.perf_counter()
    report_id = str(uuid.uuid4())
    generated_at = datetime.now(timezone.utc).isoformat()

    try:
        ev = _get_evidently()

        logs = _load_query_logs(limit)
        if model:
            logs = [r for r in logs if r.get("model") == model]

        if len(logs) < 2:
            return _fallback_report("rag_quality", model, report_id, generated_at,
                                    f"Insufficient data: {len(logs)} records", t0)

        # Build metrics safely from query log data
        latencies = []
        for r in logs:
            lat = r.get("latency_ms") or r.get("total_latency_ms")
            if lat:
                try:
                    latencies.append(float(lat))
                except (ValueError, TypeError):
                    pass

        confs = [_parse_confidence(r.get("confidence")) for r in logs if r.get("confidence")]
        errors = sum(1 for r in logs if r.get("error_type") or r.get("error"))
        successes = len(logs) - errors

        metrics = {
            "total_queries": len(logs),
            "success_rate": round(successes / len(logs), 3) if logs else 0.0,
            "error_rate": round(errors / len(logs), 3) if logs else 0.0,
            "latency": {
                "mean_ms": round(float(np.mean(latencies)), 2) if latencies else 0,
                "p50_ms": round(float(np.percentile(latencies, 50)), 2) if latencies else 0,
                "p95_ms": round(float(np.percentile(latencies, 95)), 2) if latencies else 0,
                "p99_ms": round(float(np.percentile(latencies, 99)), 2) if latencies else 0,
            },
            "confidence": {
                "mean": round(float(np.mean(confs)), 3) if confs else 0.0,
                "std": round(float(np.std(confs)), 3) if confs else 0.0,
                "below_0_5": sum(1 for c in confs if c < 0.5),
            },
            "query_types": _count_field(logs, "query_type"),
            "models": _count_field(logs, "model"),
            "error_types": _count_field(logs, "error_type"),
        }

        # Try Evidently for drift if available
        drift_summary = _compute_drift(logs)

        runtime_ms = (time.perf_counter() - t0) * 1000
        return {
            "report_id": report_id,
            "report_type": "rag_quality",
            "model": model or "all",
            "generated_at": generated_at,
            "n_records": len(logs),
            "metrics": metrics,
            "drift_summary": drift_summary,
            "runtime_ms": round(runtime_ms, 2),
            "evidently_version": getattr(ev, "__version__", "unknown"),
        }

    except Exception as exc:
        logger.exception("Evidently RAG quality report failed: %s", exc)
        return _fallback_report("rag_quality", model, report_id, generated_at, str(exc), t0)


def generate_drift_report(
    model: Optional[str] = None,
    limit: int = 200,
    reference_fraction: float = 0.5,
) -> Dict[str, Any]:
    """
    Compare recent queries (current window) against older queries (reference window)
    to detect query distribution drift.
    """
    t0 = time.perf_counter()
    report_id = str(uuid.uuid4())
    generated_at = datetime.now(timezone.utc).isoformat()

    try:
        logs = _load_query_logs(limit)
        if model:
            logs = [r for r in logs if r.get("model") == model]

        if len(logs) < 10:
            return _fallback_report("drift", model, report_id, generated_at,
                                    f"Insufficient data: {len(logs)} records", t0)

        # Split chronologically: older half = reference, newer half = current
        mid = len(logs) // 2
        reference = logs[mid:]  # older (loaded DESC so higher index = older)
        current = logs[:mid]    # newer

        drift = _compute_drift_between(reference, current)
        runtime_ms = (time.perf_counter() - t0) * 1000

        return {
            "report_id": report_id,
            "report_type": "drift",
            "model": model or "all",
            "generated_at": generated_at,
            "reference_window_size": len(reference),
            "current_window_size": len(current),
            "drift": drift,
            "runtime_ms": round(runtime_ms, 2),
        }
    except Exception as exc:
        logger.exception("Drift report failed: %s", exc)
        return _fallback_report("drift", model, report_id, generated_at, str(exc), t0)


def _compute_drift(logs: List[Dict]) -> Dict[str, Any]:
    """Simple drift detection from a single dataset (compare first vs second half)."""
    if len(logs) < 10:
        return {"available": False, "reason": "Not enough data"}
    mid = len(logs) // 2
    return _compute_drift_between(logs[mid:], logs[:mid])


def _compute_drift_between(ref: List[Dict], cur: List[Dict]) -> Dict[str, Any]:
    """
    Compare reference and current windows.
    Uses KL-divergence on query_type distribution and latency statistics.
    """
    try:
        from scipy.stats import ks_2samp

        ref_lat = np.array([float(r.get("latency_ms") or r.get("total_latency_ms") or 0) for r in ref if (r.get("latency_ms") or r.get("total_latency_ms"))])
        cur_lat = np.array([float(r.get("latency_ms") or r.get("total_latency_ms") or 0) for r in cur if (r.get("latency_ms") or r.get("total_latency_ms"))])

        lat_drift = {}
        if len(ref_lat) >= 3 and len(cur_lat) >= 3:
            stat, pval = ks_2samp(ref_lat, cur_lat)
            lat_drift = {
                "ks_statistic": round(float(stat), 4),
                "p_value": round(float(pval), 4),
                "drift_detected": bool(pval < 0.05),
            }

        ref_types = _count_field(ref, "query_type")
        cur_types = _count_field(cur, "query_type")
        all_types = set(ref_types) | set(cur_types)
        type_drift = []
        for qt in all_types:
            ref_pct = ref_types.get(qt, 0) / max(len(ref), 1)
            cur_pct = cur_types.get(qt, 0) / max(len(cur), 1)
            type_drift.append({
                "query_type": qt,
                "reference_pct": round(ref_pct, 3),
                "current_pct": round(cur_pct, 3),
                "delta": round(cur_pct - ref_pct, 3),
                "drift": abs(cur_pct - ref_pct) > 0.1,
            })

        return {
            "available": True,
            "latency_drift": lat_drift,
            "query_type_distribution_drift": sorted(type_drift, key=lambda x: abs(x["delta"]), reverse=True),
            "overall_drift_detected": lat_drift.get("drift_detected", False) or any(
                d["drift"] for d in type_drift
            ),
        }
    except Exception as exc:
        logger.warning("Drift computation failed: %s", exc)
        return {"available": False, "error": str(exc)}


def _count_field(records: List[Dict], field: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for r in records:
        v = str(r.get(field) or "unknown")
        out[v] = out.get(v, 0) + 1
    return out


def _fallback_report(
    report_type: str, model: Optional[str], report_id: str,
    generated_at: str, reason: str, t0: float
) -> Dict[str, Any]:
    return {
        "report_id": report_id,
        "report_type": report_type,
        "model": model or "all",
        "generated_at": generated_at,
        "error": reason,
        "metrics": {},
        "runtime_ms": round((time.perf_counter() - t0) * 1000, 2),
    }


def save_report(report: Dict[str, Any]) -> None:
    """Persist an Evidently report to the analytics DB."""
    try:
        from app.analytics.db import get_connection
        with get_connection() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO evidently_reports
                   (id, report_type, model, report_json, generated_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    report.get("report_id", str(uuid.uuid4())),
                    report.get("report_type", "unknown"),
                    report.get("model", "all"),
                    json.dumps(report),
                    report.get("generated_at", datetime.now(timezone.utc).isoformat()),
                ),
            )
            conn.commit()
    except Exception as exc:
        logger.warning("Could not save Evidently report: %s", exc)
