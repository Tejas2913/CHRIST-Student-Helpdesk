import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "CHRIST Student Helpdesk" in data["app"]

def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["vector_count"] > 0
    assert data["chunk_count"] > 0
    assert data["handbook_loaded"] is True

def test_models_endpoint():
    response = client.get("/api/models")
    assert response.status_code == 200
    models = response.json()
    assert len(models) > 0
    providers = {m["provider"] for m in models}
    assert "groq" in providers
    assert "ollama" in providers

def test_monitoring_endpoints():
    stats_res = client.get("/api/monitoring/stats")
    assert stats_res.status_code == 200
    stats = stats_res.json()
    assert "total_requests" in stats

    logs_res = client.get("/api/monitoring/logs?limit=5")
    assert logs_res.status_code == 200
    assert isinstance(logs_res.json(), list)

def test_chat_endpoint_schema():
    payload = {
        "message": "What is the attendance policy?",
        "provider": "groq",
        "top_k": 2
    }
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert len(data["sources"]) > 0
    assert "latency" in data
