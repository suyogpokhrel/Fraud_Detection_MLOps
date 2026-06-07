from pathlib import Path
import json
import requests
import pandas as pd

# Quick local test: load first row from engineered features and call internal prediction logic
repo_root = Path(__file__).resolve().parents[2]
X_path = repo_root / "data" / "engineered_features" / "X_engineered.csv"

df = pd.read_csv(X_path)

# Build payload using first row
first = df.iloc[0].to_dict()
payload = {"features": {k: float(v) for k, v in first.items()}}

print("Prepared payload with", len(payload['features']), "features")

print("Note: This script expects the FastAPI server to be running at http://localhost:8000")
try:
    r = requests.post("http://localhost:8000/predict", json=payload, timeout=10)
    print("Status:", r.status_code)
    print(r.json())
except Exception as exc:
    print("Request failed:", exc)
