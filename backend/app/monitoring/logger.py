import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

from app.core.config import OPS_LOG_PATH

logger = logging.getLogger(__name__)

class LLMOpsLogger:
    @staticmethod
    def log_request(
        query: str,
        provider: str,
        model: str,
        retrieval_latency_ms: float,
        generation_latency_ms: float,
        total_latency_ms: float,
        sources_retrieved: int,
        status: str,
        token_count: Optional[int] = None,
        error: Optional[str] = None,
        # New enriched fields
        query_type: Optional[str] = None,
        answer_preview: Optional[str] = None,
        confidence: Optional[str] = None,
        embedding_latency_ms: float = 0.0,
        rerank_latency_ms: float = 0.0,
        source_pages: Optional[List[int]] = None,
        retrieved_chunk_ids: Optional[List[str]] = None,
        request_id: Optional[str] = None,
    ) -> str:
        """Record structured request log to JSONL file and SQLite analytics DB."""
        req_id = request_id or str(uuid.uuid4())
        ts = datetime.now(timezone.utc).isoformat()

        record = {
            "request_id": req_id,
            "timestamp": ts,
            "query": query,
            "provider": provider,
            "model": model,
            "retrieval_latency_ms": round(retrieval_latency_ms, 2),
            "generation_latency_ms": round(generation_latency_ms, 2),
            "total_latency_ms": round(total_latency_ms, 2),
            "sources_retrieved": sources_retrieved,
            "token_count": token_count,
            "status": status,
            "error": error,
        }

        try:
            OPS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(OPS_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            logger.error(f"Failed to write to LLMOps log file: {e}")

        # Also persist to SQLite analytics DB
        try:
            from app.analytics.db import insert_query_log
            insert_query_log(
                request_id=req_id,
                timestamp=ts,
                query=query,
                provider=provider,
                model=model,
                query_type=query_type,
                answer_preview=answer_preview,
                confidence=confidence,
                total_latency_ms=round(total_latency_ms, 2),
                retrieval_latency_ms=round(retrieval_latency_ms, 2),
                generation_latency_ms=round(generation_latency_ms, 2),
                embedding_latency_ms=round(embedding_latency_ms, 2),
                rerank_latency_ms=round(rerank_latency_ms, 2),
                sources_retrieved=sources_retrieved,
                source_pages=source_pages or [],
                retrieved_chunk_ids=retrieved_chunk_ids or [],
                token_count=token_count,
                status=status,
                error=error,
            )
        except Exception as e:
            logger.error(f"Failed to write to analytics DB: {e}")

        return req_id

    @staticmethod
    def get_recent_logs(limit: int = 50) -> List[Dict[str, Any]]:
        """Read recent requests from the JSONL log file."""
        if not OPS_LOG_PATH.exists():
            return []

        logs = []
        try:
            with open(OPS_LOG_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            logs.append(json.loads(line))
                        except Exception:
                            continue
            return logs[-limit:][::-1]  # Return newest first
        except Exception as e:
            logger.error(f"Failed to read LLMOps logs: {e}")
            return []

    @staticmethod
    def get_stats() -> Dict[str, Any]:
        """Compute aggregate monitoring metrics."""
        logs = LLMOpsLogger.get_recent_logs(limit=500)
        total = len(logs)
        if total == 0:
            return {
                "total_requests": 0,
                "avg_total_latency_ms": 0.0,
                "avg_retrieval_latency_ms": 0.0,
                "avg_generation_latency_ms": 0.0,
                "error_rate_pct": 0.0,
                "providers": {},
            }

        total_lat = sum(l.get("total_latency_ms", 0) for l in logs)
        ret_lat = sum(l.get("retrieval_latency_ms", 0) for l in logs)
        gen_lat = sum(l.get("generation_latency_ms", 0) for l in logs)
        errors = sum(1 for l in logs if l.get("status") != "SUCCESS")

        providers: Dict[str, int] = {}
        for l in logs:
            prov = l.get("provider", "unknown")
            providers[prov] = providers.get(prov, 0) + 1

        return {
            "total_requests": total,
            "avg_total_latency_ms": round(total_lat / total, 2),
            "avg_retrieval_latency_ms": round(ret_lat / total, 2),
            "avg_generation_latency_ms": round(gen_lat / total, 2),
            "error_rate_pct": round((errors / total) * 100, 2),
            "providers": providers,
        }
