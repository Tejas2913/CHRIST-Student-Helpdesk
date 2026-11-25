import os
from pathlib import Path
from pydantic import BaseModel
from dotenv import load_dotenv

# Base backend directory
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent

# Load .env file from backend root or parent
load_dotenv(BACKEND_DIR / ".env")
load_dotenv(BACKEND_DIR.parent / ".env")

DATA_DIR = BACKEND_DIR / "data"
LOGS_DIR = BACKEND_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

PDF_PATH = DATA_DIR / "student_handbook.pdf"
FAISS_INDEX_PATH = DATA_DIR / "faiss_index.bin"
CHUNKS_PATH = DATA_DIR / "chunks.pkl"
INDEX_META_PATH = DATA_DIR / "index_meta.json"
STRUCTURED_RECORDS_PATH = DATA_DIR / "structured_records.json"
OPS_LOG_PATH = LOGS_DIR / "llmops_requests.jsonl"
EVAL_DIR = BACKEND_DIR / "eval"

# Embedding / chunking configuration
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 900
CHUNK_OVERLAP = 120
TOP_K_RESULTS = 5
RETRIEVAL_CANDIDATES = 24
RERANK_TOP_N = 6

# Increment when ingestion/chunk schema changes to force index rebuild.
INDEX_SCHEMA_VERSION = "3"

# Document defaults (can be overridden via env)
DOCUMENT_NAME = os.getenv("HANDBOOK_DOCUMENT_NAME", "CHRIST University Student Handbook 2026-27")
DOCUMENT_VERSION = os.getenv("HANDBOOK_DOCUMENT_VERSION", "2026-27")
ACADEMIC_YEAR = os.getenv("HANDBOOK_ACADEMIC_YEAR", "2026-27")

# Optional reranker model (cross-encoder). Leave empty for lightweight built-in reranking.
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "").strip()

# System Prompt for grounded answers
SYSTEM_PROMPT = """You are the CHRIST University Student Helpdesk Assistant. Answer student questions using ONLY the provided handbook evidence.

Rules:
1. Use the EVIDENCE as the primary and only source for university-specific facts, policies, numbers, dates, eligibility, procedures, fees, and contacts.
2. Do not invent or guess rules, numbers, dates, deadlines, fees, names, office details, or page numbers.
3. Answer the user's actual question directly and completely, including all parts of multi-part questions when evidence is available.
4. Preserve numerical accuracy, dates, eligibility conditions, exceptions, and qualifying statements exactly as supported by evidence.
5. Distinguish explicit facts from reasonable inferences only when clearly supported; prefer stating what the evidence shows.
6. If some requested information is present and some is missing, answer the supported parts and clearly state what could not be verified in the available handbook evidence.
7. If the evidence does not contain the answer, say clearly: "I couldn't find this information in the available university handbook/document." Then advise contacting the relevant university office if appropriate.
8. Use simple, student-friendly language and format clearly with headings/bullets/numbered steps where helpful.
9. Cite supporting sources for factual statements using the provided evidence labels/page numbers (for example: [Handbook p. 56]). Only cite sources that appear in the provided evidence.
10. Output ONLY your final direct response to the student. Do NOT output internal thoughts, chain of thought, reasoning steps, drafting reflections, or phrases such as 'Let's check' or 'Let's re-read'."""

class Settings(BaseModel):
    app_name: str = "CHRIST Student Helpdesk API"
    app_version: str = "2.1.0"
    debug: bool = False

    # Provider Settings
    default_provider: str = os.getenv("DEFAULT_PROVIDER", "groq")
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")

    # Active tested Groq Models
    default_groq_model: str = os.getenv("DEFAULT_GROQ_MODEL", "qwen/qwen3.8-27b")
    available_groq_models: list[str] = [
        "qwen/qwen3.8-27b",
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
    ]

    # Local Ollama Settings
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    default_ollama_model: str = os.getenv("DEFAULT_OLLAMA_MODEL", "qwen2.5:1.5b")
    available_ollama_models: list[str] = [
        "qwen2.5:1.5b",
        "llama3.2:1b",
        "gemma2:2b",
    ]

    # Retrieval / generation controls
    top_k_results: int = int(os.getenv("TOP_K_RESULTS", str(TOP_K_RESULTS)))
    retrieval_candidates: int = int(os.getenv("RETRIEVAL_CANDIDATES", str(RETRIEVAL_CANDIDATES)))
    rerank_top_n: int = int(os.getenv("RERANK_TOP_N", str(RERANK_TOP_N)))
    enable_reranker: bool = os.getenv("ENABLE_RERANKER", "true").lower() in {"1", "true", "yes", "on"}
    enable_multi_hop: bool = os.getenv("ENABLE_MULTI_HOP", "true").lower() in {"1", "true", "yes", "on"}

settings = Settings()
