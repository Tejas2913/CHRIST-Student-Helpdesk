import os
from fastapi import APIRouter
from app.core.config import settings, PDF_PATH
from app.rag.vectorstore import VectorStoreManager
from app.llm.factory import LLMProviderFactory
from app.schemas.chat import HealthResponse

router = APIRouter(tags=["Health"])

@router.get("/health", response_model=HealthResponse)
async def check_health():
    vs = VectorStoreManager.get_instance()
    factory = LLMProviderFactory.get_instance()
    is_active_healthy = factory.is_provider_healthy(factory.active_provider_name)

    return HealthResponse(
        status="healthy",
        app_name=settings.app_name,
        version=settings.app_version,
        vector_count=vs.index.ntotal if vs.index else 0,
        chunk_count=len(vs.chunks),
        active_provider=factory.active_provider_name,
        active_model=factory.active_model_name,
        provider_available=is_active_healthy,
        handbook_loaded=os.path.exists(PDF_PATH),
    )
