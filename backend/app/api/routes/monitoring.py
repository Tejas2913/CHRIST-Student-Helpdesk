from fastapi import APIRouter, Query
from typing import List, Dict, Any
from app.monitoring.logger import LLMOpsLogger
from app.schemas.chat import LogEntry

router = APIRouter(tags=["Monitoring"])

@router.get("/monitoring/logs", response_model=List[LogEntry])
async def get_logs(limit: int = Query(50, ge=1, le=200)):
    logs = LLMOpsLogger.get_recent_logs(limit=limit)
    return [LogEntry(**l) for l in logs]

@router.get("/monitoring/stats")
async def get_stats() -> Dict[str, Any]:
    return LLMOpsLogger.get_stats()
