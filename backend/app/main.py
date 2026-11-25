from contextlib import asynccontextmanager
import logging
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.core.config import settings, PDF_PATH, OPS_LOG_PATH
from app.rag.vectorstore import VectorStoreManager
from app.api.routes import health, models, chat, monitoring
from app.api.routes import analytics as analytics_route
from app.api.routes import explainability as xai_route
from app.analytics.db import init_db, migrate_jsonl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing CHRIST Student Helpdesk vector store on startup...")
    try:
        VectorStoreManager.get_instance()
        logger.info("CHRIST Student Helpdesk backend initialized successfully.")
    except Exception:
        logger.exception("Failed to initialize vector store on startup")
        # Keep API running so health/errors are visible; retrieval endpoints will retry.

    # Initialize analytics SQLite DB and migrate existing JSONL logs
    try:
        init_db()
        migrated = migrate_jsonl(OPS_LOG_PATH)
        if migrated:
            logger.info("Migrated %d JSONL records into analytics DB", migrated)
    except Exception:
        logger.exception("Failed to initialize analytics DB")
    yield
    logger.info("Shutting down CHRIST Student Helpdesk backend.")

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Full-Stack AI-Powered CHRIST University Student Helpdesk Chatbot API",
    lifespan=lifespan,
)

# CORS setup for React / Vite frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(health.router, prefix="/api")
app.include_router(models.router, prefix="/api")
app.include_router(chat.router, prefix="/api")
app.include_router(monitoring.router, prefix="/api")
app.include_router(analytics_route.router, prefix="/api")
app.include_router(xai_route.router, prefix="/api")


@app.get("/api/handbook")
async def get_handbook_pdf():
    if not os.path.exists(PDF_PATH):
        raise HTTPException(status_code=404, detail="Handbook PDF not found")
    return FileResponse(
        path=str(PDF_PATH),
        filename="student_handbook.pdf",
        media_type="application/pdf",
    )

@app.get("/")
def root():
    return {
        "app": settings.app_name,
        "version": settings.app_version,
        "docs_url": "/docs",
        "health_check": "/api/health",
        "handbook_pdf": "/api/handbook",
    }
