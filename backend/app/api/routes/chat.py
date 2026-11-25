from fastapi import APIRouter, HTTPException
from app.schemas.chat import ChatRequest, ChatResponse
from app.rag.pipeline import RAGPipeline

router = APIRouter(tags=["Chat"])
pipeline = RAGPipeline()


def _friendly_error(e: Exception) -> str:
    """Convert raw API or runtime errors into student-friendly messages."""
    msg = str(e)
    if "model_not_found" in msg or "does not exist" in msg:
        return (
            "The selected model is not available on this Groq account. "
            "Please switch to a different model in the sidebar (e.g. qwen/qwen3.8-27b)."
        )
    if "AuthenticationError" in msg or "Invalid API key" in msg:
        return (
            "Groq API key is invalid or not configured. "
            "Please set a valid GROQ_API_KEY in backend/.env and restart the server."
        )
    if "RateLimitError" in msg or "rate_limit" in msg:
        return "Groq rate limit reached. Please wait a moment and try again."
    if "All attempted Groq models failed" in msg:
        return (
            "Could not reach any Groq model. "
            "Please check your API key and ensure at least one model is available on your account."
        )
    return f"Chat processing failed: {msg}"


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        response = pipeline.process_chat(request)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=_friendly_error(e))
