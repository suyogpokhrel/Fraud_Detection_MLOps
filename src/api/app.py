from pathlib import Path
import pickle
import json
from typing import Dict

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

repo_root = Path(__file__).resolve().parents[2]
MODEL_DIR = repo_root / "models" / "trained_models"
FEATURE_LIST = repo_root / "data" / "engineered_features" / "all_features.txt"


def load_model_and_metadata():
    model_path = MODEL_DIR / "best_model.pkl"
    scaler_path = MODEL_DIR / "scaler.pkl"
    if not model_path.exists() or not scaler_path.exists():
        raise FileNotFoundError("Trained model or scaler not found in models/trained_models")

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    if not FEATURE_LIST.exists():
        raise FileNotFoundError("Feature list not found: data/engineered_features/all_features.txt")

    feature_names = [ln.strip() for ln in FEATURE_LIST.read_text().splitlines() if ln.strip()]

    return model, scaler, feature_names


model, scaler, FEATURE_NAMES = load_model_and_metadata()

app = FastAPI(title="Fraud Detection API")


class PredictionRequest(BaseModel):
    features: Dict[str, float]


@app.post("/predict")
def predict(req: PredictionRequest):
    # Ensure all required features provided
    missing = [f for f in FEATURE_NAMES if f not in req.features]
    if missing:
        raise HTTPException(status_code=400, detail={"missing_features": missing})

    # Build ordered feature vector
    x = np.array([req.features[f] for f in FEATURE_NAMES], dtype=float).reshape(1, -1)

    # Scale and predict
    try:
        x_scaled = scaler.transform(x)
        proba = float(model.predict_proba(x_scaled)[0, 1])
        label = int(model.predict(x_scaled)[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"prediction": label, "probability": proba}


@app.get("/features")
def features():
    return {"feature_names": FEATURE_NAMES}
