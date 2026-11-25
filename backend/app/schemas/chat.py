from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import time

class SourceCitation(BaseModel):
    chunk_id: int
    page: int
    score: float
    text: str
    # Extended metadata (optional for backward compatibility)
    id: Optional[str] = None
    document: Optional[str] = None
    document_version: Optional[str] = None
    academic_year: Optional[str] = None
    section: Optional[str] = None
    subsection: Optional[str] = None
    clause: Optional[str] = None
    content_type: Optional[str] = None
    table_name: Optional[str] = None
    campus: Optional[str] = None
    programme: Optional[str] = None

class ChatMessage(BaseModel):
    role: str
    content: str
    timestamp: Optional[float] = Field(default_factory=time.time)

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="Student query or question")
    provider: Optional[str] = Field(None, description="'groq' or 'ollama'")
    model: Optional[str] = Field(None, description="Model identifier")
    top_k: Optional[int] = Field(None, ge=1, le=10, description="Number of source chunks to retrieve")
    conversation_history: Optional[List[ChatMessage]] = Field(default_factory=list)
    # Optional metadata filters
    campus: Optional[str] = None
    programme: Optional[str] = None
    academic_year: Optional[str] = None
    section: Optional[str] = None
    content_type: Optional[str] = None

class LatencyMetrics(BaseModel):
    retrieval_ms: float
    generation_ms: float
    total_ms: float

class RetrievalDebug(BaseModel):
    query_type: Optional[str] = None
    confidence: Optional[str] = None
    confidence_reason: Optional[str] = None
    resolved_query: Optional[str] = None
    subqueries: List[str] = Field(default_factory=list)
    filters: Dict[str, Any] = Field(default_factory=dict)
    retrieval_strategy: List[str] = Field(default_factory=list)
    evidence_count: int = 0

class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceCitation]
    provider: str
    model: str
    latency: LatencyMetrics
    timestamp: float = Field(default_factory=time.time)
    error: Optional[str] = None
    confidence: Optional[str] = None
    query_type: Optional[str] = None
    debug: Optional[RetrievalDebug] = None

class ModelInfo(BaseModel):
    id: str
    name: str
    provider: str
    is_available: bool
    is_default: bool
    notes: Optional[str] = None

class ModelSelectionRequest(BaseModel):
    provider: str
    model: str

class HealthResponse(BaseModel):
    status: str
    app_name: str
    version: str
    vector_count: int
    chunk_count: int
    active_provider: str
    active_model: str
    provider_available: bool = True
    handbook_loaded: bool

class LogEntry(BaseModel):
    request_id: str
    timestamp: str
    provider: str
    model: str
    query: str
    retrieval_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float
    sources_retrieved: int
    status: str
    error: Optional[str] = None
