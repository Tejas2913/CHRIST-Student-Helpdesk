"""
Benchmark runner comparing SHAP and LIME across evaluation benchmark questions.
Exports results to CSV and JSON for academic/technical evaluation reporting.
"""
import csv
import json
import logging
import sys
from pathlib import Path

# Ensure backend root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.explainability.shap_explainer import get_shap_explainer
from app.explainability.lime_explainer import get_lime_explainer
from app.explainability.explanation_evaluator import run_comparison
from app.rag.vectorstore import VectorStoreManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BENCHMARK_PATH = Path(__file__).resolve().parent / "benchmark.json"
OUTPUT_CSV_PATH = Path(__file__).resolve().parent / "xai_benchmark_results.csv"


def run_benchmark():
    logger.info("Loading benchmark queries from %s", BENCHMARK_PATH)
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    queries = data.get("queries", data) if isinstance(data, dict) else data
    vs = VectorStoreManager()
    shap_exp = get_shap_explainer()
    lime_exp = get_lime_explainer()

    fieldnames = [
        "query_id",
        "query",
        "retrieved_chunks_count",
        "top_chunk_score",
        "shap_runtime_ms",
        "lime_runtime_ms",
        "spearman_rank_correlation",
        "shap_sparsity",
        "lime_sparsity",
        "lime_local_r2",
    ]

    rows = []
    logger.info("Executing XAI evaluations across %d benchmark queries...", len(queries))
    for idx, q_item in enumerate(queries, 1):
        q_text = q_item.get("query") or q_item.get("question") or str(q_item)
        chunks = vs.hybrid_search(q_text, top_k=3)

        shap_res = shap_exp.explain_retrieval(q_text, chunks)
        lime_res = lime_exp.explain_retrieval(q_text, chunks)
        cmp = run_comparison(q_text, shap_res, lime_res, model=None)

        top_score = chunks[0].get("score", 0.0) if chunks else 0.0

        row = {
            "query_id": idx,
            "query": q_text,
            "retrieved_chunks_count": len(chunks),
            "top_chunk_score": round(float(top_score), 4),
            "shap_runtime_ms": round(float(shap_res.get("runtime_ms", 0.0)), 2),
            "lime_runtime_ms": round(float(lime_res.get("runtime_ms", 0.0)), 2),
            "spearman_rank_correlation": round(float(cmp["metrics"]["feature_consistency"]["value"]), 4),
            "shap_sparsity": round(float(cmp["metrics"]["sparsity"]["shap"]), 4),
            "lime_sparsity": round(float(cmp["metrics"]["sparsity"]["lime"]), 4),
            "lime_local_r2": cmp["metrics"]["local_fidelity"]["lime"],
        }
        rows.append(row)
        logger.info("[%d/%d] Evaluated query: %s", idx, len(queries), q_text[:40])

    with open(OUTPUT_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Successfully exported XAI benchmark results to %s", OUTPUT_CSV_PATH)
    return rows


if __name__ == "__main__":
    run_benchmark()
