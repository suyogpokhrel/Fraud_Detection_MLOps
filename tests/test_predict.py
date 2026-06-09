"""
Unit tests for model prediction pipeline.
Tests loading, feature engineering, and prediction correctness.
"""
import pytest
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
MODELS_DIR = repo_root / "models" / "trained_models"

@pytest.fixture(scope="module")
def model():
    return joblib.load(MODELS_DIR / "best_model.pkl")

@pytest.fixture(scope="module")
def scaler():
    return joblib.load(MODELS_DIR / "scaler.pkl")

@pytest.fixture(scope="module")
def feature_names():
    return json.loads((MODELS_DIR / "feature_names.json").read_text())

@pytest.fixture(scope="module")
def sample_data():
    X = pd.read_csv(repo_root / "data" / "engineered_features" / "X_engineered.csv")
    y = pd.read_csv(repo_root / "data" / "engineered_features" / "y_engineered.csv").squeeze()
    return X, y

def test_model_loads():
    m = joblib.load(MODELS_DIR / "best_model.pkl")
    assert m is not None

def test_scaler_loads():
    s = joblib.load(MODELS_DIR / "scaler.pkl")
    assert s is not None

def test_feature_names_count(feature_names):
    assert len(feature_names) == 20

def test_feature_names_no_old_columns(feature_names):
    forbidden = ["unusuallogin", "isFlaggedFraud", "branch_fraud_rate",
                 "acct_type_Current", "acct_type_Savings"]
    for col in forbidden:
        assert col not in feature_names, f"Old fake column found: {col}"

def test_model_predicts(model, sample_data, feature_names):
    X, y = sample_data
    preds = model.predict(X.head(100))
    assert len(preds) == 100
    assert set(preds).issubset({0, 1})

def test_model_predict_proba(model, sample_data):
    X, y = sample_data
    probs = model.predict_proba(X.head(100))[:, 1]
    assert len(probs) == 100
    assert probs.min() >= 0.0
    assert probs.max() <= 1.0

def test_fraud_recall(model, sample_data):
    X, y = sample_data
    preds = model.predict(X)
    fraud_idx = y == 1
    recall = preds[fraud_idx].sum() / fraud_idx.sum()
    assert recall > 0.95, f"Recall too low: {recall:.4f}"

def test_metrics_file_exists():
    assert (MODELS_DIR / "metrics.json").exists()

def test_metadata_file_exists():
    assert (MODELS_DIR / "metadata.json").exists()

def test_metadata_best_model():
    meta = json.loads((MODELS_DIR / "metadata.json").read_text())
    assert "best_model" in meta
    assert meta["best_model"] in ["XGBoost", "Random Forest", "Logistic Regression"]

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
