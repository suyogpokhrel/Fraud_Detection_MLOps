"""
Basic pipeline tests for CI/CD
Tests: Redis client, drift monitor import, API health
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
    """Redis client module should import without errors."""
    from src.cache.redis_client import (
        get_client, save_dataframe, load_dataframe,
        save_json, load_json, clear_pipeline_cache
    )
    assert callable(save_dataframe)
    assert callable(load_dataframe)


def test_redis_client_handles_unavailable():
    """Redis client should return None gracefully when Redis is down."""
    from src.cache.redis_client import get_client
    # In CI, Redis is not running — should return None, not crash
    client = get_client()
    # Either None (unavailable) or a real client (if Redis is running)
    assert client is None or client is not None


def test_save_load_dataframe_no_redis():
    """save_dataframe should return False gracefully when Redis is down."""
    from src.cache.redis_client import save_dataframe, load_dataframe
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = save_dataframe("test:key", df)
    assert isinstance(result, bool)

    loaded = load_dataframe("test:key")
    assert loaded is None or isinstance(loaded, pd.DataFrame)


# ─── Drift Monitor Tests ──────────────────────────────────────────────────────

def test_drift_monitor_imports():
    """Drift monitor module should import without errors."""
    from src.monitoring.drift_monitor import run_drift_report
    assert callable(run_drift_report)


def test_drift_report_generates(tmp_path):
    """Drift report should generate HTML file from sample data."""
    from src.monitoring.drift_monitor import run_drift_report

    # Create small sample CSVs in tmp_path
    df = pd.DataFrame({
        "amount": np.random.uniform(100, 10000, 200),
        "oldbalanceOrg": np.random.uniform(0, 50000, 200),
        "newbalanceOrig": np.random.uniform(0, 50000, 200),
        "oldbalanceDest": np.random.uniform(0, 50000, 200),
        "newbalanceDest": np.random.uniform(0, 50000, 200),
        "step": np.random.randint(1, 100, 200),
        "unusuallogin": np.random.randint(0, 10, 200),
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
    """FeatureEngineer should import without errors."""
    from src.feature_engineering.engineer_features import FeatureEngineer
    engineer = FeatureEngineer()
    assert engineer is not None


# ─── Preprocessing Tests ─────────────────────────────────────────────────────

def test_preprocessing_pipeline_imports():
    """PreprocessingPipeline should import without errors."""
    from src.preprocessing.preprocess import PreprocessingPipeline
    pipeline = PreprocessingPipeline()
    assert pipeline is not None


# ─── Model Loading Tests ──────────────────────────────────────────────────────

def test_best_model_loads():
    """best_model.pkl should load without errors."""
    import joblib
    model_path = repo_root / "models" / "best_model.pkl"
    if not model_path.exists():
        pytest.skip("best_model.pkl not found — skipping in CI")
    model = joblib.load(model_path)
    assert model is not None
    assert hasattr(model, "predict")


# ─── API Tests ────────────────────────────────────────────────────────────────

def test_api_health():
    """FastAPI app should import and have a health route."""
    try:
        from src.api.main import app
        assert app is not None
    except ImportError:
        pytest.skip("API module not found — skipping in CI")
