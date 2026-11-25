"""
SQLite-backed analytics database for the CHRIST Student Helpdesk dashboard.
Uses stdlib sqlite3 only — no extra dependencies required.
"""

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# DB lives next to the FAISS index
DB_PATH = Path(__file__).resolve().parents[3] / "data" / "analytics.db"


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_DDL = """
CREATE TABLE IF NOT EXISTS query_logs (
    id                      TEXT PRIMARY KEY,
    timestamp               TEXT NOT NULL,
    query                   TEXT NOT NULL,
    provider                TEXT,
    model                   TEXT,
    query_type              TEXT,
    answer_preview          TEXT,
    confidence              TEXT,
    total_latency_ms        REAL,
    retrieval_latency_ms    REAL,
    generation_latency_ms   REAL,
    embedding_latency_ms    REAL,
    rerank_latency_ms       REAL,
    sources_retrieved       INTEGER,
    source_pages            TEXT,       -- JSON array of ints
    retrieved_chunk_ids     TEXT,       -- JSON array of strings
    token_count             INTEGER,
    status                  TEXT,
    error                   TEXT,
    eval_correctness        REAL,
    eval_faithfulness       REAL,
    eval_relevance          REAL
);

CREATE INDEX IF NOT EXISTS idx_timestamp   ON query_logs (timestamp);
CREATE INDEX IF NOT EXISTS idx_model       ON query_logs (model);
CREATE INDEX IF NOT EXISTS idx_status      ON query_logs (status);
CREATE INDEX IF NOT EXISTS idx_query_type  ON query_logs (query_type);

CREATE TABLE IF NOT EXISTS explanations (
    id                      TEXT PRIMARY KEY,
    query_id                TEXT,
    model                   TEXT,
    explainer_type          TEXT,
    prediction_target       TEXT,
    input_features          TEXT,
    feature_contributions   TEXT,
    shap_values             TEXT,
    lime_weights            TEXT,
    runtime_ms              REAL,
    faithfulness_score      REAL,
    stability_score         REAL,
    sparsity_score          REAL,
    local_fidelity_score    REAL,
    timestamp               TEXT,
    model_version           TEXT,
    explainer_config        TEXT
);

CREATE TABLE IF NOT EXISTS attributions (
    id                      TEXT PRIMARY KEY,
    query_id                TEXT,
    model                   TEXT,
    sentence                TEXT,
    attributed_chunk_id     TEXT,
    attributed_page         INTEGER,
    attributed_section      TEXT,
    supporting_passage      TEXT,
    attribution_confidence  REAL,
    is_supported            INTEGER,
    timestamp               TEXT
);

CREATE TABLE IF NOT EXISTS evidently_reports (
    id                      TEXT PRIMARY KEY,
    report_type             TEXT,
    model                   TEXT,
    report_json             TEXT,
    generated_at            TEXT,
    date_range_from         TEXT,
    date_range_to           TEXT
);

CREATE INDEX IF NOT EXISTS idx_expl_query_id ON explanations (query_id);
CREATE INDEX IF NOT EXISTS idx_expl_type     ON explanations (explainer_type);
CREATE INDEX IF NOT EXISTS idx_attr_query_id ON attributions (query_id);
"""


# ---------------------------------------------------------------------------
# Connection helper
# ---------------------------------------------------------------------------

def get_connection() -> sqlite3.Connection:
    """Public alias for _get_conn — used by explainability module."""
    return _get_conn()


def _get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create schema if not already present."""
    with _get_conn() as conn:
        conn.executescript(_DDL)
    logger.info("Analytics DB initialised at %s", DB_PATH)


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

def insert_query_log(
    *,
    request_id: str,
    timestamp: str,
    query: str,
    provider: str,
    model: str,
    query_type: Optional[str] = None,
    answer_preview: Optional[str] = None,
    confidence: Optional[str] = None,
    total_latency_ms: float = 0.0,
    retrieval_latency_ms: float = 0.0,
    generation_latency_ms: float = 0.0,
    embedding_latency_ms: float = 0.0,
    rerank_latency_ms: float = 0.0,
    sources_retrieved: int = 0,
    source_pages: Optional[List[int]] = None,
    retrieved_chunk_ids: Optional[List[str]] = None,
    token_count: Optional[int] = None,
    status: str = "SUCCESS",
    error: Optional[str] = None,
) -> None:
    sql = """
    INSERT OR REPLACE INTO query_logs (
        id, timestamp, query, provider, model,
        query_type, answer_preview, confidence,
        total_latency_ms, retrieval_latency_ms, generation_latency_ms,
        embedding_latency_ms, rerank_latency_ms,
        sources_retrieved, source_pages, retrieved_chunk_ids,
        token_count, status, error
    ) VALUES (
        :id, :timestamp, :query, :provider, :model,
        :query_type, :answer_preview, :confidence,
        :total_latency_ms, :retrieval_latency_ms, :generation_latency_ms,
        :embedding_latency_ms, :rerank_latency_ms,
        :sources_retrieved, :source_pages, :retrieved_chunk_ids,
        :token_count, :status, :error
    )
    """
    params = {
        "id": request_id,
        "timestamp": timestamp,
        "query": query,
        "provider": provider,
        "model": model,
        "query_type": query_type,
        "answer_preview": answer_preview,
        "confidence": confidence,
        "total_latency_ms": total_latency_ms,
        "retrieval_latency_ms": retrieval_latency_ms,
        "generation_latency_ms": generation_latency_ms,
        "embedding_latency_ms": embedding_latency_ms,
        "rerank_latency_ms": rerank_latency_ms,
        "sources_retrieved": sources_retrieved,
        "source_pages": json.dumps(source_pages or []),
        "retrieved_chunk_ids": json.dumps(retrieved_chunk_ids or []),
        "token_count": token_count,
        "status": status,
        "error": error,
    }
    try:
        with _get_conn() as conn:
            conn.execute(sql, params)
    except Exception as e:
        logger.error("Failed to insert query log: %s", e)


# ---------------------------------------------------------------------------
# Migrate existing JSONL → SQLite (run once on startup)
# ---------------------------------------------------------------------------

def migrate_jsonl(jsonl_path: Path) -> int:
    """Import all JSONL log entries into SQLite (skips existing IDs)."""
    if not jsonl_path.exists():
        return 0

    imported = 0
    with _get_conn() as conn:
        with open(jsonl_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # check if already present
                exists = conn.execute(
                    "SELECT 1 FROM query_logs WHERE id = ?", (rec.get("request_id", ""),)
                ).fetchone()
                if exists:
                    continue

                conn.execute(
                    """
                    INSERT OR IGNORE INTO query_logs (
                        id, timestamp, query, provider, model,
                        total_latency_ms, retrieval_latency_ms, generation_latency_ms,
                        sources_retrieved, token_count, status, error
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        rec.get("request_id", ""),
                        rec.get("timestamp", ""),
                        rec.get("query", ""),
                        rec.get("provider", ""),
                        rec.get("model", ""),
                        rec.get("total_latency_ms", 0.0),
                        rec.get("retrieval_latency_ms", 0.0),
                        rec.get("generation_latency_ms", 0.0),
                        rec.get("sources_retrieved", 0),
                        rec.get("token_count"),
                        rec.get("status", "SUCCESS"),
                        rec.get("error"),
                    ),
                )
                imported += 1

    logger.info("Migrated %d JSONL records into analytics DB", imported)
    return imported


# ---------------------------------------------------------------------------
# Read helpers
# ---------------------------------------------------------------------------

def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    d = dict(row)
    for field in ("source_pages", "retrieved_chunk_ids"):
        if isinstance(d.get(field), str):
            try:
                d[field] = json.loads(d[field])
            except Exception:
                d[field] = []
    return d


def get_summary() -> Dict[str, Any]:
    """KPI summary across all logs."""
    with _get_conn() as conn:
        total = conn.execute("SELECT COUNT(*) FROM query_logs").fetchone()[0]
        if total == 0:
            return {
                "total_queries": 0,
                "queries_today": 0,
                "total_responses": 0,
                "avg_latency_ms": 0.0,
                "avg_retrieval_ms": 0.0,
                "avg_generation_ms": 0.0,
                "error_rate_pct": 0.0,
                "best_model": None,
                "models_active": 0,
            }

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        queries_today = conn.execute(
            "SELECT COUNT(*) FROM query_logs WHERE timestamp LIKE ?", (f"{today}%",)
        ).fetchone()[0]

        agg = conn.execute(
            """
            SELECT
                AVG(total_latency_ms)      AS avg_total,
                AVG(retrieval_latency_ms)  AS avg_ret,
                AVG(generation_latency_ms) AS avg_gen,
                SUM(CASE WHEN status != 'SUCCESS' THEN 1 ELSE 0 END) AS errors,
                COUNT(*) AS cnt
            FROM query_logs
            """
        ).fetchone()

        successes = conn.execute(
            "SELECT COUNT(*) FROM query_logs WHERE status = 'SUCCESS'"
        ).fetchone()[0]

        # Best model by lowest avg total latency (among successful)
        best_row = conn.execute(
            """
            SELECT model, AVG(total_latency_ms) AS avg_lat
            FROM query_logs
            WHERE status = 'SUCCESS'
            GROUP BY model
            ORDER BY avg_lat ASC
            LIMIT 1
            """
        ).fetchone()

        models_active = conn.execute(
            "SELECT COUNT(DISTINCT model) FROM query_logs"
        ).fetchone()[0]

        return {
            "total_queries": total,
            "queries_today": queries_today,
            "total_responses": successes,
            "avg_latency_ms": round(agg["avg_total"] or 0, 2),
            "avg_retrieval_ms": round(agg["avg_ret"] or 0, 2),
            "avg_generation_ms": round(agg["avg_gen"] or 0, 2),
            "error_rate_pct": round((agg["errors"] / agg["cnt"]) * 100, 2) if agg["cnt"] else 0.0,
            "best_model": best_row["model"] if best_row else None,
            "models_active": models_active,
        }


def get_model_stats() -> List[Dict[str, Any]]:
    """Per-model aggregated statistics."""
    with _get_conn() as conn:
        rows = conn.execute(
            """
            SELECT
                model,
                provider,
                COUNT(*)                                                AS total_queries,
                AVG(total_latency_ms)                                   AS avg_latency_ms,
                AVG(retrieval_latency_ms)                               AS avg_retrieval_ms,
                AVG(generation_latency_ms)                              AS avg_generation_ms,
                SUM(CASE WHEN status = 'SUCCESS' THEN 1 ELSE 0 END)    AS successes,
                SUM(CASE WHEN status != 'SUCCESS' THEN 1 ELSE 0 END)   AS errors,
                AVG(sources_retrieved)                                  AS avg_sources,
                AVG(token_count)                                        AS avg_tokens
            FROM query_logs
            GROUP BY model
            ORDER BY total_queries DESC
            """
        ).fetchall()

        result = []
        for r in rows:
            d = dict(r)
            total = d["total_queries"] or 1
            d["success_rate_pct"] = round((d["successes"] / total) * 100, 2)
            d["error_rate_pct"] = round((d["errors"] / total) * 100, 2)

            # Compute p50 / p95 from individual rows for this model
            latencies = [
                row[0]
                for row in conn.execute(
                    "SELECT total_latency_ms FROM query_logs WHERE model = ? ORDER BY total_latency_ms",
                    (d["model"],),
                ).fetchall()
            ]
            if latencies:
                n = len(latencies)
                d["p50_latency_ms"] = round(latencies[int(n * 0.50)], 2)
                d["p95_latency_ms"] = round(latencies[min(int(n * 0.95), n - 1)], 2)
            else:
                d["p50_latency_ms"] = 0.0
                d["p95_latency_ms"] = 0.0

            for key in ("avg_latency_ms", "avg_retrieval_ms", "avg_generation_ms", "avg_sources", "avg_tokens"):
                if d[key] is not None:
                    d[key] = round(d[key], 2)

            result.append(d)

        return result


def get_query_logs(
    *,
    page: int = 1,
    limit: int = 50,
    model: Optional[str] = None,
    status: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    search: Optional[str] = None,
    query_type: Optional[str] = None,
    sort_by: str = "timestamp",
    sort_dir: str = "desc",
) -> Tuple[List[Dict[str, Any]], int]:
    """Return paginated query log rows + total count."""
    conditions = []
    params: List[Any] = []

    if model:
        conditions.append("model = ?")
        params.append(model)
    if status:
        conditions.append("status = ?")
        params.append(status)
    if date_from:
        conditions.append("timestamp >= ?")
        params.append(date_from)
    if date_to:
        conditions.append("timestamp <= ?")
        params.append(date_to + "T23:59:59")
    if search:
        conditions.append("query LIKE ?")
        params.append(f"%{search}%")
    if query_type:
        conditions.append("query_type = ?")
        params.append(query_type)

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    allowed_sort = {"timestamp", "total_latency_ms", "retrieval_latency_ms", "generation_latency_ms"}
    sort_col = sort_by if sort_by in allowed_sort else "timestamp"
    direction = "DESC" if sort_dir.lower() == "desc" else "ASC"

    with _get_conn() as conn:
        total = conn.execute(f"SELECT COUNT(*) FROM query_logs {where}", params).fetchone()[0]
        offset = (page - 1) * limit
        rows = conn.execute(
            f"SELECT * FROM query_logs {where} ORDER BY {sort_col} {direction} LIMIT ? OFFSET ?",
            params + [limit, offset],
        ).fetchall()

    return [_row_to_dict(r) for r in rows], total


def get_query_detail(request_id: str) -> Optional[Dict[str, Any]]:
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM query_logs WHERE id = ?", (request_id,)
        ).fetchone()
    return _row_to_dict(row) if row else None


def get_volume_over_time(group_by: str = "day") -> List[Dict[str, Any]]:
    """Query volume grouped by hour / day / week."""
    if group_by == "hour":
        fmt = "%Y-%m-%dT%H:00"
    elif group_by == "week":
        fmt = "%Y-W%W"
    else:
        fmt = "%Y-%m-%d"

    with _get_conn() as conn:
        rows = conn.execute(
            f"""
            SELECT
                strftime('{fmt}', timestamp) AS period,
                model,
                COUNT(*) AS count
            FROM query_logs
            GROUP BY period, model
            ORDER BY period ASC
            """
        ).fetchall()

    return [dict(r) for r in rows]


def get_latency_trend(group_by: str = "day") -> List[Dict[str, Any]]:
    if group_by == "hour":
        fmt = "%Y-%m-%dT%H:00"
    else:
        fmt = "%Y-%m-%d"

    with _get_conn() as conn:
        rows = conn.execute(
            f"""
            SELECT
                strftime('{fmt}', timestamp)  AS period,
                model,
                AVG(total_latency_ms)         AS avg_latency_ms,
                AVG(retrieval_latency_ms)     AS avg_retrieval_ms,
                AVG(generation_latency_ms)    AS avg_generation_ms,
                COUNT(*)                      AS count
            FROM query_logs
            GROUP BY period, model
            ORDER BY period ASC
            """
        ).fetchall()

    return [
        {**dict(r), "avg_latency_ms": round(r["avg_latency_ms"] or 0, 2),
         "avg_retrieval_ms": round(r["avg_retrieval_ms"] or 0, 2),
         "avg_generation_ms": round(r["avg_generation_ms"] or 0, 2)}
        for r in rows
    ]


def get_category_distribution() -> List[Dict[str, Any]]:
    with _get_conn() as conn:
        rows = conn.execute(
            """
            SELECT COALESCE(query_type, 'unknown') AS query_type, COUNT(*) AS count
            FROM query_logs
            GROUP BY query_type
            ORDER BY count DESC
            """
        ).fetchall()
    return [dict(r) for r in rows]


def get_error_breakdown() -> List[Dict[str, Any]]:
    with _get_conn() as conn:
        rows = conn.execute(
            """
            SELECT model, status, COUNT(*) AS count
            FROM query_logs
            GROUP BY model, status
            ORDER BY model, status
            """
        ).fetchall()
    return [dict(r) for r in rows]


def export_csv_rows(
    *,
    model: Optional[str] = None,
    status: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> List[Dict[str, Any]]:
    rows, _ = get_query_logs(
        page=1, limit=10000,
        model=model, status=status,
        date_from=date_from, date_to=date_to,
    )
    return rows
