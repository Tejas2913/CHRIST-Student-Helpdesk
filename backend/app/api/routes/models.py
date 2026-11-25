from fastapi import APIRouter, HTTPException
from typing import List
from app.llm.factory import LLMProviderFactory
from app.schemas.chat import ModelInfo, ModelSelectionRequest

router = APIRouter(tags=["Models"])

@router.get("/models", response_model=List[ModelInfo])
async def list_models():
    factory = LLMProviderFactory.get_instance()
    models = factory.list_all_models()
    return [ModelInfo(**m) for m in models]

@router.post("/models/select")
async def select_model(request: ModelSelectionRequest):
    factory = LLMProviderFactory.get_instance()
    try:
        factory.set_active_selection(request.provider, request.model)
        return {
            "success": True,
            "message": f"Active model set to {request.provider} ({request.model})",
            "active_provider": factory.active_provider_name,
            "active_model": factory.active_model_name,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
