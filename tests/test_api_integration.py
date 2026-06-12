"""
Integration tests for FastAPI prediction endpoints.
"""
import pytest
import json
from fastapi.testclient import TestClient
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent
sys.path.insert(0, str(repo_root))

from src.api.fastapi_app import app

client = TestClient(app)

# ── Sample transactions ───────────────────────────────────
LEGIT_TX = {
    "step": 1, "type": "PAYMENT", "amount": 1000.0,
    "oldbalanceOrg": 50000.0, "newbalanceOrig": 49000.0,
    "oldbalanceDest": 0.0, "newbalanceDest": 1000.0
}

FRAUD_TX = {
    "step": 1, "type": "TRANSFER", "amount": 170136.0,
    "oldbalanceOrg": 170136.0, "newbalanceOrig": 0.0,
    "oldbalanceDest": 0.0, "newbalanceDest": 0.0
}

# ── Basic endpoint tests ──────────────────────────────────
def test_root():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"
    assert r.json()["model_loaded"] is True

def test_model_info():
    r = client.get("/model-info")
    assert r.status_code == 200
    data = r.json()
    assert "best_model" in data
    assert "feature_count" in data

# ── Predict endpoint tests ────────────────────────────────
def test_predict_returns_correct_fields():
    r = client.post("/predict", json=LEGIT_TX)
    assert r.status_code == 200
    data = r.json()
    assert "is_fraud" in data
    assert "fraud_probability" in data
    assert "risk_level" in data

def test_predict_is_fraud_binary():
    r = client.post("/predict", json=LEGIT_TX)
    assert r.json()["is_fraud"] in [0, 1]

def test_predict_probability_range():
    r = client.post("/predict", json=LEGIT_TX)
    prob = r.json()["fraud_probability"]
    assert 0.0 <= prob <= 1.0

def test_predict_risk_level_valid():
    r = client.post("/predict", json=LEGIT_TX)
    assert r.json()["risk_level"] in ["LOW", "MEDIUM", "HIGH"]

def test_predict_fraud_transaction():
    r = client.post("/predict", json=FRAUD_TX)
    assert r.status_code == 200
    data = r.json()
    assert data["is_fraud"] == 1
    assert data["fraud_probability"] > 0.5

# ── Batch predict tests ───────────────────────────────────
def test_batch_predict():
    payload = {"transactions": [LEGIT_TX, FRAUD_TX]}
    r = client.post("/batch_predict", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["total"] == 2
    assert len(data["predictions"]) == 2

def test_batch_predict_fraud_count():
    payload = {"transactions": [LEGIT_TX, FRAUD_TX]}
    r = client.post("/batch_predict", json=payload)
    data = r.json()
    assert data["fraud_count"] >= 1

def test_batch_predict_fields():
    payload = {"transactions": [LEGIT_TX]}
    r = client.post("/batch_predict", json=payload)
    pred = r.json()["predictions"][0]
    assert "is_fraud" in pred
    assert "fraud_probability" in pred
    assert "risk_level" in pred

# ── Metrics endpoint ──────────────────────────────────────
def test_metrics_endpoint():
    r = client.get("/metrics")
    assert r.status_code == 200
    data = r.json()
    assert "total_requests" in data
    assert "fraud_detected" in data
    assert "fraud_rate_live" in data

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
