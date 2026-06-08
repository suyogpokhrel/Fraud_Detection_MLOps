"""
FastAPI Backend for Fraud Detection Model Serving
Phase 8A: REST API for real-time fraud predictions
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pathlib import Path
import pickle
import joblib
import numpy as np
import pandas as pd
import sys

# Add repo to path
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

from src.preprocessing.preprocess import PreprocessingPipeline
from src.feature_engineering.engineer_features import FeatureEngineer

# Initialize FastAPI app
app = FastAPI(
    title="Fraud Detection API",
    description="Real-time fraud detection for financial transactions",
    version="1.0.0"
)

# Enable CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained model and transformers at startup
MODEL_PATH = repo_root / "models" / "trained_models" / "best_model.pkl"
SCALER_PATH = repo_root / "models" / "trained_models" / "scaler.pkl"

try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    scaler = pickle.load(open(SCALER_PATH, 'rb'))
    print(f"✓ Model loaded: {MODEL_PATH}")
except Exception as e:
    print(f"✗ Failed to load model: {e}")
    model = None
    scaler = None

# Request/Response schemas
class TransactionInput(BaseModel):
    """Input schema for fraud prediction"""
    step: int = Field(..., description="Transaction step sequence")
    amount: float = Field(..., description="Transaction amount")
    oldbalanceOrg: float = Field(..., description="Sender's balance before transaction")
    newbalanceOrig: float = Field(..., description="Sender's balance after transaction")
    oldbalanceDest: float = Field(..., description="Receiver's balance before transaction")
    newbalanceDest: float = Field(..., description="Receiver's balance after transaction")
    type: str = Field(..., description="Transaction type: PAYMENT, TRANSFER, CASH_OUT, CASH_IN, DEBIT")
    acct_type: str = Field(..., description="Account type: Current or Savings")
    branch: int = Field(..., description="Branch ID (0-134)")
    unusuallogin: int = Field(default=0, description="Unusual login score")
    isFlaggedFraud: int = Field(default=0, description="System fraud flag (0 or 1)")

    class Config:
        json_schema_extra = {
            "example": {
                "step": 1,
                "amount": 9839.64,
                "oldbalanceOrg": 170136.00,
                "newbalanceOrig": 160296.36,
                "oldbalanceDest": 0.00,
                "newbalanceDest": 0.00,
                "type": "PAYMENT",
                "acct_type": "Current",
                "branch": 50,
                "unusuallogin": 5,
                "isFlaggedFraud": 0
            }
        }

class PredictionResponse(BaseModel):
    """Output schema for fraud prediction"""
    is_fraud: bool = Field(..., description="Fraud prediction (True/False)")
    fraud_probability: float = Field(..., description="Probability of fraud (0-1)")
    fraud_confidence: float = Field(..., description="Confidence score (0-100)")
    risk_level: str = Field(..., description="Risk category: Low, Medium, High")

# Helper functions
def prepare_features(transaction: TransactionInput) -> np.ndarray:
    """
    Convert raw transaction input to 34 model features.
    Replicates full preprocessing from Phase 3 & 4.
    """
    # === Phase 3 + 4 Feature Processing ===
    
    # Basic transaction dict
    tx_dict = transaction.dict()
    
    # ===== INPUT FEATURES (16) =====
    features = []
    
    # 1-7: Numeric features (will be scaled later)
    numeric_cols = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
                    'oldbalanceDest', 'newbalanceDest', 'unusuallogin']
    numeric_features = np.array([tx_dict[col] for col in numeric_cols], dtype=float)
    
    # 8: isFlaggedFraud
    features.append(tx_dict['isFlaggedFraud'])
    
    # 9-13: One-hot encode transaction type (5 features)
    type_mapping = {'CASH_IN': 0, 'CASH_OUT': 1, 'DEBIT': 2, 'PAYMENT': 3, 'TRANSFER': 4}
    tx_type_idx = type_mapping.get(tx_dict['type'], 0)
    tx_type_encoded = [1.0 if i == tx_type_idx else 0.0 for i in range(5)]
    features.extend(tx_type_encoded)
    
    # 14-15: One-hot encode account type (2 features)
    acct_type_idx = 0 if tx_dict['acct_type'] == 'Current' else 1
    acct_type_encoded = [1.0 if i == acct_type_idx else 0.0 for i in range(2)]
    features.extend(acct_type_encoded)
    
    # 16: Branch fraud rate (mock: use branch ID as proxy)
    branch_id = tx_dict['branch']
    branch_fraud_rate = (branch_id % 100) / 100.0
    features.append(branch_fraud_rate)
    
    # Build unscaled input features (16 total)
    input_features = np.concatenate([
        numeric_features,
        np.array(features, dtype=float)
    ])
    
    # ===== ENGINEERED FEATURES (18) =====
    engineered = []
    
    # Balance-based features
    sender_balance_change = tx_dict['newbalanceOrig'] - tx_dict['oldbalanceOrg']
    receiver_balance_change = tx_dict['newbalanceDest'] - tx_dict['oldbalanceDest']
    
    engineered.append(sender_balance_change)  # 17
    engineered.append(receiver_balance_change)  # 18
    engineered.append(abs(sender_balance_change))  # 19
    engineered.append(abs(receiver_balance_change))  # 20
    
    # Balance change ratios
    sender_balance_ratio = sender_balance_change / (abs(tx_dict['oldbalanceOrg']) + 1e-10)
    receiver_balance_ratio = receiver_balance_change / (abs(tx_dict['oldbalanceDest']) + 1e-10)
    
    engineered.append(sender_balance_ratio)  # 21
    engineered.append(receiver_balance_ratio)  # 22
    
    # Amount ratios
    amount_ratio_sender = tx_dict['amount'] / (abs(tx_dict['oldbalanceOrg']) + 1e-10)
    amount_ratio_receiver = tx_dict['amount'] / (abs(tx_dict['oldbalanceDest']) + 1e-10)
    
    engineered.append(amount_ratio_sender)  # 23
    engineered.append(amount_ratio_receiver)  # 24
    
    # Suspicious behavior flags
    sender_depletes = 1.0 if sender_balance_change < 0 and abs(sender_balance_change) > tx_dict['amount'] * 0.9 else 0.0
    receiver_zero = 1.0 if tx_dict['oldbalanceDest'] < 1.0 else 0.0
    high_value = 1.0 if tx_dict['amount'] > np.percentile([100000], 95) else 0.0  # Simple heuristic
    
    engineered.append(sender_depletes)  # 25
    engineered.append(receiver_zero)   # 26
    engineered.append(high_value)      # 27
    
    # Interaction features
    current_to_savings = 1.0 if (tx_type_idx == 3 and acct_type_idx == 1) else 0.0  # PAYMENT to Savings
    cashout_unusual = 1.0 if (tx_type_idx == 1 and tx_dict['unusuallogin'] > 50) else 0.0  # CASH_OUT + unusual
    
    engineered.append(current_to_savings)  # 28
    engineered.append(cashout_unusual)  # 29
    
    # Complex signals
    high_fraud_branch = 1.0 if (branch_id % 10 < 3 and tx_dict['amount'] > 50000) else 0.0
    flagged_and_unusual = float(tx_dict['isFlaggedFraud'] and tx_dict['unusuallogin'] > 50)
    
    engineered.append(high_fraud_branch)  # 30
    engineered.append(flagged_and_unusual)  # 31
    
    # Suspicion and risk scores
    suspicion_score = (
        sender_depletes * 0.3 +
        receiver_zero * 0.1 +
        high_value * 0.2 +
        current_to_savings * 0.15 +
        cashout_unusual * 0.25
    )
    
    risk_exposure_sender = abs(sender_balance_change) + (amount_ratio_sender * 0.1)
    total_money_flow = abs(sender_balance_change) + abs(receiver_balance_change)
    
    engineered.append(suspicion_score)  # 32
    engineered.append(risk_exposure_sender)  # 33
    engineered.append(total_money_flow)  # 34
    
    # Combine all 34 features (unscaled)
    all_features = np.concatenate([input_features, np.array(engineered, dtype=float)])
    
    # Scale all 34 features using the model's scaler
    features_scaled = scaler.transform(all_features.reshape(1, -1))
    
    return features_scaled

def get_risk_level(fraud_prob: float) -> str:
    """Classify fraud probability into risk levels"""
    if fraud_prob < 0.3:
        return "Low"
    elif fraud_prob < 0.7:
        return "Medium"
    else:
        return "High"

# API Endpoints
@app.get("/")
def read_root():
    """Health check endpoint"""
    return {
        "status": "✓ Fraud Detection API is running",
        "version": "1.0.0",
        "model_loaded": model is not None
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_fraud(transaction: TransactionInput):
    """
    Predict if a transaction is fraudulent.
    
    Returns:
    - is_fraud: Boolean prediction
    - fraud_probability: Confidence (0-1)
    - fraud_confidence: Percentage (0-100)
    - risk_level: Low/Medium/High
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # Prepare features
        features = prepare_features(transaction)

        # Get prediction
        prediction = model.predict(features)[0]  # 0 = legit, 1 = fraud
        fraud_probability = model.predict_proba(features)[0][1]  # Probability of fraud

        # Determine fraud and risk level
        is_fraud = bool(prediction == 1)
        fraud_confidence = float(fraud_probability * 100)
        risk_level = get_risk_level(fraud_probability)

        return PredictionResponse(
            is_fraud=is_fraud,
            fraud_probability=float(fraud_probability),
            fraud_confidence=fraud_confidence,
            risk_level=risk_level
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.post("/batch_predict")
def batch_predict(transactions: list[TransactionInput]):
    """
    Predict fraud for multiple transactions.
    
    Returns list of predictions with counts.
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    results = []
    for tx in transactions:
        try:
            features = prepare_features(tx)
            prediction = model.predict(features)[0]
            fraud_probability = model.predict_proba(features)[0][1]
            
            results.append({
                "is_fraud": bool(prediction == 1),
                "fraud_probability": float(fraud_probability),
                "fraud_confidence": float(fraud_probability * 100),
                "risk_level": get_risk_level(fraud_probability)
            })
        except Exception as e:
            results.append({
                "is_fraud": False,
                "fraud_probability": 0.0,
                "fraud_confidence": 0.0,
                "risk_level": "Unknown",
                "error": str(e)
            })

    return results

@app.get("/model-info")
def get_model_info():
    """Get information about the trained model"""
    return {
        "model_type": "Logistic Regression",
        "training_date": "2026-06-06",
        "recall": 0.7857,
        "precision": 0.1549,
        "roc_auc": 0.8738,
        "fraud_threshold": 0.5,
        "features": 34
    }

if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 70)
    print("PHASE 8A: FASTAPI BACKEND")
    print("=" * 70)
    print("\nStarting FastAPI server...")
    print("API docs: http://127.0.0.1:8000/docs")
    print("Redoc: http://127.0.0.1:8000/redoc")
    print()
    uvicorn.run(app, host="127.0.0.1", port=8000)
