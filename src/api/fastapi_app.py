"""
Phase 7: FastAPI Prediction API
"""
from pathlib import Path
import sys
import json
import time
import joblib
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional
import uvicorn

repo_root = Path(__file__).resolve().parent
sys.path.insert(0, str(repo_root))

app = FastAPI(
    title="Fraud Detection API",
    description="MLOps fraud detection using PaySim dataset",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

metrics_store = {
    "total_requests": 0,
    "total_predictions": 0,
    "fraud_detected": 0,
    "errors": 0,
    "total_response_ms": 0.0,
}

@app.middleware("http")
async def monitor(request: Request, call_next):
    start = time.time()
    try:
        response = await call_next(request)
        metrics_store["total_requests"] += 1
        metrics_store["total_response_ms"] += (time.time() - start) * 1000
        return response
    except Exception:
        metrics_store["errors"] += 1
        raise

MODELS_DIR = repo_root / "models" / "trained_models"

def load_artifacts():
    model  = joblib.load(MODELS_DIR / "best_model.pkl")
    scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    feature_names = json.loads((MODELS_DIR / "feature_names.json").read_text())
    meta_path = MODELS_DIR / "metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return model, scaler, feature_names, meta

try:
    model, scaler, feature_names, model_meta = load_artifacts()
    print(f"✓ Model loaded: {model_meta.get('best_model', 'unknown')}")
except Exception as e:
    print(f"✗ Model load failed: {e}")
    model, scaler, feature_names, model_meta = None, None, None, {}

class Transaction(BaseModel):
    step:           int   = Field(..., example=1)
    type:           str   = Field(..., example="TRANSFER")
    amount:         float = Field(..., example=9839.64)
    oldbalanceOrg:  float = Field(..., example=170136.0)
    newbalanceOrig: float = Field(..., example=160296.36)
    oldbalanceDest: float = Field(..., example=0.0)
    newbalanceDest: float = Field(..., example=0.0)

class BatchRequest(BaseModel):
    transactions: List[Transaction]

def prepare(transactions: list) -> pd.DataFrame:
    rows = [t.model_dump() for t in transactions]
    df = pd.DataFrame(rows)

    # Step 1: one-hot encode type (matches preprocess.py)
    for t in ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]:
        df[f"type_{t}"] = (df["type"] == t).astype(int)

    # Step 2: engineered features (matches engineer_features.py)
    df["sender_balance_change"]    = df["oldbalanceOrg"]  - df["newbalanceOrig"]
    df["receiver_balance_change"]  = df["newbalanceDest"] - df["oldbalanceDest"]
    df["amount_ratio_sender"]      = np.where(
        df["oldbalanceOrg"] > 0, df["amount"] / df["oldbalanceOrg"], 0)
    df["amount_ratio_receiver"]    = np.where(
        df["oldbalanceDest"] > 0, df["amount"] / df["oldbalanceDest"], 0)
    df["sender_depletes_account"]  = (
        (df["oldbalanceOrg"] > 0) & (df["newbalanceOrig"] == 0)).astype(int)
    df["receiver_had_zero_balance"] = (df["oldbalanceDest"] == 0).astype(int)
    df["is_transfer_or_cashout"]   = df["type"].isin(["TRANSFER","CASH_OUT"]).astype(int)
    df["balance_mismatch_sender"]  = abs(df["sender_balance_change"] - df["amount"])
    df["balance_mismatch_receiver"]= abs(df["receiver_balance_change"] - df["amount"])

    # Step 3: drop non-numeric columns
    df = df.drop(columns=["type"], errors="ignore")

    # Step 4: align to exact feature order from training
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names]

    # Step 5: scale only the 6 numeric cols the scaler was fitted on
    numeric_cols = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
                    'oldbalanceDest', 'newbalanceDest']
    df[numeric_cols] = scaler.transform(df[numeric_cols])
    return df

@app.get("/")
def root():
    return {"status": "ok", "service": "Fraud Detection API", "version": "2.0.0"}

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.get("/model-info")
def model_info():
    return {
        "best_model":    model_meta.get("best_model", "unknown"),
        "feature_count": model_meta.get("feature_count", "unknown"),
        "fraud_rate":    model_meta.get("fraud_rate_full", "unknown"),
        "best_recall":   model_meta.get("best_recall", "unknown"),
        "best_f1":       model_meta.get("best_f1", "unknown"),
    }

@app.post("/predict")
def predict(tx: Transaction):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        df   = prepare([tx])
        pred = int(model.predict(df)[0])
        prob = float(model.predict_proba(df)[0][1])
        metrics_store["total_predictions"] += 1
        if pred == 1:
            metrics_store["fraud_detected"] += 1
        return {
            "is_fraud":           pred,
            "fraud_probability":  round(prob, 4),
            "risk_level":         "HIGH" if prob > 0.7 else "MEDIUM" if prob > 0.4 else "LOW"
        }
    except Exception as e:
        metrics_store["errors"] += 1
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch_predict")
def batch_predict(req: BatchRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        df    = prepare(req.transactions)
        preds = model.predict(df).tolist()
        probs = model.predict_proba(df)[:, 1].tolist()
        metrics_store["total_predictions"] += len(preds)
        metrics_store["fraud_detected"]    += sum(preds)
        return {
            "predictions": [
                {
                    "is_fraud":          int(p),
                    "fraud_probability": round(pr, 4),
                    "risk_level":        "HIGH" if pr > 0.7 else "MEDIUM" if pr > 0.4 else "LOW"
                }
                for p, pr in zip(preds, probs)
            ],
            "total":       len(preds),
            "fraud_count": sum(preds)
        }
    except Exception as e:
        metrics_store["errors"] += 1
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
def get_metrics():
    total = metrics_store["total_predictions"]
    return {
        **metrics_store,
        "fraud_rate_live":  round(metrics_store["fraud_detected"] / total, 4) if total else 0,
        "avg_response_ms":  round(metrics_store["total_response_ms"] /
                            max(metrics_store["total_requests"], 1), 2),
    }

if __name__ == "__main__":
    uvicorn.run("fastapi_app:app", host="0.0.0.0", port=8000, reload=True)
