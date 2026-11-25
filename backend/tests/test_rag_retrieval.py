import pytest
from app.rag.vectorstore import VectorStoreManager

def test_vector_store_initialization():
    vs = VectorStoreManager.get_instance()
    assert vs.index is not None
    assert vs.index.ntotal > 0
    assert len(vs.chunks) > 0

def test_retrieval_attendance_query():
    vs = VectorStoreManager.get_instance()
    results = vs.search("What is the minimum attendance requirement?", top_k=3)
    assert len(results) > 0
    assert any("attendance" in r["text"].lower() for r in results)
    assert "page" in results[0]
    assert "score" in results[0]

def test_retrieval_leave_policy_query():
    vs = VectorStoreManager.get_instance()
    results = vs.search("What is the leave application procedure?", top_k=3)
    assert len(results) > 0
    assert any("leave" in r["text"].lower() or "student services" in r["text"].lower() for r in results)
