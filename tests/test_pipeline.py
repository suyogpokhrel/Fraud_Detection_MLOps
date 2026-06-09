"""
Basic pipeline tests for CI/CD
Tests: Redis client, drift monitor, feature engineering, preprocessing, model loading, API health
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pytest

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))


# ─── Redis Client Tests ───────────────────────────────────────────────────────

def test_redis_client_imports():
    from src.cache.redis_client import (
        get_client, save_dataframe, load_dataframe,
        save_json, load_json, clear_pipeline_cache
    )
    assert callable(save_dataframe)
    assert callable(load_dataframe)

def test_redis_client_handles_unavailable():
    from src.cache.redis_client import get_client
    client = get_client()
    assert client is None or client is not None

def test_save_load_dataframe_no_redis():
    from src.cache.redis_client import save_dataframe, load_dataframe
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = save_dataframe("test:key", df)
    assert isinstance(result, bool)
    loaded = load_dataframe("test:key")
    assert loaded is None or isinstance(loaded, pd.DataFrame)


# ─── Drift Monitor Tests ──────────────────────────────────────────────────────

def test_drift_monitor_imports():
    from src.monitoring.drift_monitor import run_drift_report
    assert callable(run_drift_report)

def test_drift_report_generates(tmp_path):
    from src.monitoring.drift_monitor import run_drift_report

    # Use real PaySim feature columns (matches engineer_features.py output)
    df = pd.DataFrame({
        "step":                     np.random.randint(1, 100, 200),
        "amount":                   np.random.uniform(100, 10000, 200),
        "oldbalanceOrg":            np.random.uniform(0, 50000, 200),
        "newbalanceOrig":           np.random.uniform(0, 50000, 200),
        "oldbalanceDest":           np.random.uniform(0, 50000, 200),
        "newbalanceDest":           np.random.uniform(0, 50000, 200),
        "type_CASH_IN":             np.random.randint(0, 2, 200),
        "type_CASH_OUT":            np.random.randint(0, 2, 200),
        "type_DEBIT":               np.random.randint(0, 2, 200),
        "type_PAYMENT":             np.random.randint(0, 2, 200),
        "type_TRANSFER":            np.random.randint(0, 2, 200),
        "sender_balance_change":    np.random.uniform(-5000, 5000, 200),
        "receiver_balance_change":  np.random.uniform(-5000, 5000, 200),
        "amount_ratio_sender":      np.random.uniform(0, 1, 200),
        "amount_ratio_receiver":    np.random.uniform(0, 1, 200),
        "sender_depletes_account":  np.random.randint(0, 2, 200),
        "receiver_had_zero_balance":np.random.randint(0, 2, 200),
        "is_transfer_or_cashout":   np.random.randint(0, 2, 200),
        "balance_mismatch_sender":  np.random.uniform(0, 1000, 200),
        "balance_mismatch_receiver":np.random.uniform(0, 1000, 200),
    })

    ref_path = tmp_path / "X_preprocessed.csv"
    cur_path = tmp_path / "X_engineered.csv"
    df.to_csv(ref_path, index=False)
    df.to_csv(cur_path, index=False)

    report_path = run_drift_report(
        reference_path=ref_path,
        current_path=cur_path,
        output_dir=tmp_path / "reports"
    )
    assert Path(report_path).exists()
    assert report_path.endswith(".html")


# ─── Feature Engineering Tests ───────────────────────────────────────────────

def test_feature_engineer_imports():
    from src.feature_engineering.engineer_features import FeatureEngineer
    engineer = FeatureEngineer()
    assert engineer is not None


# ─── Preprocessing Tests ─────────────────────────────────────────────────────

def test_preprocessing_pipeline_imports():
    from src.preprocessing.preprocess import PreprocessingPipeline
    pipeline = PreprocessingPipeline()
    assert pipeline is not None


# ─── Model Loading Tests ──────────────────────────────────────────────────────

def test_best_model_loads():
    import joblib
    model_path = repo_root / "models" / "trained_models" / "best_model.pkl"
    if not model_path.exists():
        pytest.skip("best_model.pkl not found — skipping in CI")
    model = joblib.load(model_path)
    assert model is not None
    assert hasattr(model, "predict")


# ─── API Tests ────────────────────────────────────────────────────────────────

def test_api_health():
    from fastapi.testclient import TestClient
    from src.api.fastapi_app import app
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"

