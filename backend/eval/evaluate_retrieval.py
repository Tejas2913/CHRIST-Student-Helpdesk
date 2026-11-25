import json
import time
from statistics import mean
from pathlib import Path
import sys

# Run from backend/: python eval/evaluate_retrieval.py
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.rag.vectorstore import VectorStoreManager
from app.rag.query_analysis import analyze_query

BENCH = Path(__file__).resolve().parent / "benchmark.json"
KS = [1, 3, 5]


def reciprocal_rank(pages, expected_pages):
    exp = set(expected_pages or [])
    if not exp:
        return None
    for i, p in enumerate(pages, 1):
        if p in exp:
            return 1.0 / i
    return 0.0


def main():
    data = json.loads(BENCH.read_text(encoding="utf-8"))
    vs = VectorStoreManager.get_instance()
    results = []
    for item in data:
        q = item["question"]
        expected = item.get("expected_pages") or []
        a = analyze_query(q)
        resolved = a.get("resolved_query") or q
        base = resolved.split("\nRelated terms:")[0].split("\n[Retrieval hints]:")[0].strip() or q
        t0 = time.perf_counter()
        docs = vs.hybrid_search(base, top_k=max(KS))
        ms = (time.perf_counter() - t0) * 1000
        pages = [d.get("page") for d in docs]
        kw = [k.lower() for k in (item.get("expected_keywords") or [])]
        text_blob = " ".join((d.get("text") or "") for d in docs).lower()
        kw_hits = sum(1 for k in kw if k in text_blob) if kw else None
        row = {
            "id": item["id"],
            "category": item.get("category"),
            "question": q,
            "expected_pages": expected,
            "retrieved_pages": pages,
            "retrieval_ms": round(ms, 2),
        }
        for k in KS:
            topk = pages[:k]
            if expected:
                hit = any(p in set(expected) for p in topk)
                recall = 1.0 if hit else 0.0
                prec = (1.0 / k) if hit else 0.0
            else:
                hit = None
                recall = None
                prec = None
            row[f"hit@{k}"] = hit
            row[f"recall@{k}"] = recall
            row[f"precision@{k}"] = prec
        row["mrr"] = reciprocal_rank(pages, expected)
        if kw_hits is not None:
            row["keyword_coverage"] = round(kw_hits / max(len(kw), 1), 3)
        results.append(row)

    ans = [r for r in results if r["expected_pages"]]
    summary = {"total": len(results), "answerable": len(ans), "unanswerable": len(results) - len(ans)}
    for k in KS:
        vals = [r[f"hit@{k}"] for r in ans if r[f"hit@{k}"] is not None]
        pvals = [r[f"precision@{k}"] for r in ans if r[f"precision@{k}"] is not None]
        summary[f"hit_rate@{k}"] = round(mean(vals), 3) if vals else None
        summary[f"recall@{k}"] = round(mean(vals), 3) if vals else None
        summary[f"precision@{k}"] = round(mean(pvals), 3) if pvals else None
    mrrs = [r["mrr"] for r in ans if r["mrr"] is not None]
    summary["mrr"] = round(mean(mrrs), 3) if mrrs else None
    summary["avg_retrieval_ms"] = round(mean(r["retrieval_ms"] for r in results), 2)
    summary["unanswerable_note"] = "Unanswerable items have no gold pages; generation-layer abstention/confidence is evaluated separately."

    report = {"summary": summary, "results": results}
    out = Path(__file__).resolve().parent / "retrieval_eval_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
