"""
Comprehensive tests for Explainable AI (XAI), SHAP, LIME, Attribution, and Evidently.
"""
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.explainability.shap_explainer import get_shap_explainer, RETRIEVAL_FEATURE_NAMES
from app.explainability.lime_explainer import get_lime_explainer
from app.explainability.rag_attribution import get_attributor
from app.explainability.explanation_evaluator import (
    measure_faithfulness,
    measure_stability,
    measure_sparsity,
    measure_feature_consistency,
    run_comparison,
)
from app.explainability.evidently_monitor import (
    generate_rag_quality_report,
    generate_drift_report,
)

client = TestClient(app)

MOCK_CHUNKS = [
    {
        "chunk_id": 101,
        "page": 42,
        "section": "ATTENDANCE REGULATIONS",
        "subsection": "Minimum Requirement",
        "document": "Student Handbook 2026-27",
        "text": "Students must have a minimum aggregate attendance of 85% in each course to be eligible for End Semester Examinations.",
        "search_text": "Students must have a minimum aggregate attendance of 85% in each course to be eligible for End Semester Examinations.",
        "score": 0.89,
        "content_type": "eligibility",
    },
    {
        "chunk_id": 102,
        "page": 43,
        "section": "CONDONATION",
        "subsection": "Medical Exemption",
        "document": "Student Handbook 2026-27",
        "text": "Condonation of shortage of attendance between 75% and 84% may be granted on valid medical grounds.",
        "search_text": "Condonation of shortage of attendance between 75% and 84% may be granted on valid medical grounds.",
        "score": 0.72,
        "content_type": "policy_regulation",
    },
]


def test_shap_retrieval_explanation():
    explainer = get_shap_explainer()
    res = explainer.explain_retrieval(
        query="What is the minimum attendance required?",
        chunks=MOCK_CHUNKS,
    )
    assert res["target"] == "retrieval_relevance"
    assert "global_importance" in res
    assert len(res["chunk_explanations"]) == len(MOCK_CHUNKS)
    assert res["runtime_ms"] >= 0
    for name in RETRIEVAL_FEATURE_NAMES:
        assert name in res["global_importance"]


def test_shap_intent_explanation():
    explainer = get_shap_explainer()
    res = explainer.explain_intent("What is the minimum attendance requirement?")
    assert res["target"] == "intent_classification"
    assert "predicted_class" in res
    assert "confidence" in res
    assert res["confidence"] > 0


def test_lime_intent_explanation():
    explainer = get_lime_explainer()
    res = explainer.explain_intent("What is the minimum attendance requirement?")
    assert res["target"] == "intent_classification"
    assert "word_importance" in res
    assert len(res["word_importance"]) > 0
    assert "runtime_ms" in res


def test_lime_retrieval_explanation():
    explainer = get_lime_explainer()
    res = explainer.explain_retrieval(
        query="What is the minimum attendance required?",
        chunks=MOCK_CHUNKS,
    )
    assert res["target"] == "retrieval_relevance"
    assert "feature_importance" in res
    assert len(res["feature_importance"]) > 0


def test_rag_sentence_attribution():
    attributor = get_attributor()
    answer = (
        "Students must maintain at least 85% attendance to write the exams. "
        "The campus library is open 24 hours every day of the week."
    )
    res = attributor.attribute(answer=answer, chunks=MOCK_CHUNKS)
    assert "sentence_attributions" in res
    assert len(res["sentence_attributions"]) == 2
    assert "coverage" in res
    # First sentence should match attendance chunk
    s1 = res["sentence_attributions"][0]
    assert s1["support_level"] in ("supported", "partial")
    assert s1["attributed_page"] == 42


def test_evaluation_metrics():
    # Sparsity
    weights = {"feat1": 0.5, "feat2": 0.001, "feat3": 0.0001, "feat4": 0.8}
    sp = measure_sparsity(weights)
    assert 0.0 <= sp <= 1.0

    # Consistency
    shap_w = {"a": 0.8, "b": 0.4, "c": 0.1}
    lime_w = {"a": 0.7, "b": 0.3, "c": 0.05}
    corr = measure_feature_consistency(shap_w, lime_w)
    assert corr > 0.8


def test_evidently_reports():
    q_rep = generate_rag_quality_report(limit=50)
    assert "report_id" in q_rep
    assert q_rep["report_type"] == "rag_quality"

    d_rep = generate_drift_report(limit=50)
    assert "report_id" in d_rep
    assert d_rep["report_type"] == "drift"


def test_api_xai_shap_endpoint():
    res = client.post("/api/xai/shap", json={
        "query": "What is the minimum attendance requirement?",
        "target": "retrieval_relevance",
        "chunks": MOCK_CHUNKS,
    })
    assert res.status_code == 200
    data = res.json()
    assert data["target"] == "retrieval_relevance"
    assert "global_importance" in data


def test_api_xai_lime_endpoint():
    res = client.post("/api/xai/lime", json={
        "query": "What is the minimum attendance requirement?",
        "target": "intent_classification",
    })
    assert res.status_code == 200
    data = res.json()
    assert "word_importance" in data


def test_api_xai_attribution_endpoint():
    res = client.post("/api/xai/attribution", json={
        "query": "What is the attendance rule?",
        "answer": "Attendance required is 85% in each course.",
        "chunks": MOCK_CHUNKS,
    })
    assert res.status_code == 200
    data = res.json()
    assert "sentence_attributions" in data
    assert "coverage" in data


def test_api_xai_compare_endpoint():
    res = client.post("/api/xai/compare", json={
        "query": "What is the minimum attendance?",
        "target": "retrieval_relevance",
        "chunks": MOCK_CHUNKS,
    })
    assert res.status_code == 200
    data = res.json()
    assert "metrics" in data
    assert "summary" in data


def test_api_evidently_endpoints():
    res = client.post("/api/evidently/generate", json={
        "report_type": "rag_quality",
        "limit": 50,
    })
    assert res.status_code == 200
    assert "report_id" in res.json()

    rep_list = client.get("/api/evidently/reports")
    assert rep_list.status_code == 200
    assert isinstance(rep_list.json(), list)
