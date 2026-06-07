# Phase 8: API Deployment Guide

## Overview

This phase deploys the fraud detection model via a REST API (FastAPI) + Interactive Dashboard (Streamlit) with Docker containerization.

**Architecture:**
```
┌─────────────────────────────────────────────────────┐
│         Docker Network (fraud_network)              │
│                                                     │
│  ┌──────────────────────────────────────────────┐  │
│  │  Streamlit UI Container (Port 8501)          │  │
│  │  - Single transaction prediction              │  │
│  │  - Batch CSV analysis                         │  │
│  │  - Model info & explanation                   │  │
│  └──────────────────────────────────────────────┘  │
│                      │                              │
│                      │ HTTP Requests                │
│                      ▼                              │
│  ┌──────────────────────────────────────────────┐  │
│  │  FastAPI Backend Container (Port 8000)       │  │
│  │  - /predict - Single prediction               │  │
│  │  - /batch_predict - Bulk processing           │  │
│  │  - /model-info - Model metadata               │  │
│  │                                               │  │
│  │  Loaded Models: logistic_regression.pkl      │  │
│  │  Loaded Scaler: scaler.pkl                   │  │
│  └──────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

## Quick Start (Local Development)

### Prerequisites
- Python 3.11+
- conda environment: `fraud_mlops`
- MariaDB (for training only, not needed for deployment)

### Step 1: Start FastAPI Backend
```bash
cd /home/suyog/Desktop/A_space/MLOPS/Final_Project/Fraud_Detection_MLOps
python src/api/fastapi_app.py
```

**Expected Output:**
```
✓ Model loaded: /path/to/logistic_regression.pkl
Starting FastAPI server...
INFO: Started server process [PID]
INFO: Application startup complete.
INFO: Uvicorn running on http://127.0.0.1:8000
```

**Verify API:**
- Health check: `curl http://127.0.0.1:8000/`
- API docs: http://127.0.0.1:8000/docs
- ReDoc: http://127.0.0.1:8000/redoc

### Step 2: Start Streamlit Frontend (New Terminal)
```bash
cd /home/suyog/Desktop/A_space/MLOPS/Final_Project/Fraud_Detection_MLOps
streamlit run src/ui/streamlit_app.py
```

**Expected Output:**
```
          You can now view your Streamlit app in your browser.
          Local URL: http://localhost:8501
          Network URL: http://[YOUR_IP]:8501
```

**Access UI:** http://localhost:8501

### Step 3: Test Prediction
1. Open http://localhost:8501
2. Go to "🔍 Single Prediction"
3. Fill form with sample transaction:
   - Amount: 9839.64
   - Sender Balance: 170136.00
   - Type: PAYMENT
   - etc.
4. Click "🚀 Predict"
5. View result with fraud probability and risk level

## Docker Deployment

### Prerequisites
- Docker installed
- Docker Compose installed

### Build Images
```bash
docker-compose build
```

### Run Services
```bash
docker-compose up -d
```

**Check Status:**
```bash
docker-compose ps
```

**View Logs:**
```bash
# FastAPI logs
docker logs fraud_detection_api -f

# Streamlit logs
docker logs fraud_detection_ui -f
```

**Access Services:**
- API: http://localhost:8000
- UI: http://localhost:8501
- API Docs: http://localhost:8000/docs

### Stop Services
```bash
docker-compose down
```

## API Endpoints

### 1. Health Check
```bash
GET /
```
Returns: `{"status": "API is running"}`

### 2. Single Prediction
```bash
POST /predict
```
**Request:**
```json
{
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
```

**Response:**
```json
{
  "is_fraud": false,
  "fraud_probability": 0.1234,
  "fraud_confidence": 12.34,
  "risk_level": "Low"
}
```

### 3. Batch Prediction
```bash
POST /batch_predict
```
**Request:** Array of transaction objects
```json
[
  { ...transaction1... },
  { ...transaction2... }
]
```

**Response:** Array of predictions
```json
[
  { "is_fraud": false, "fraud_probability": 0.1234, ... },
  { "is_fraud": true, "fraud_probability": 0.8756, ... }
]
```

### 4. Model Info
```bash
GET /model-info
```
**Response:**
```json
{
  "model_type": "Logistic Regression",
  "recall": 0.7857,
  "precision": 0.1549,
  "roc_auc": 0.8738,
  "fraud_threshold": 0.5,
  "features": 34,
  "training_date": "2024-01-15"
}
```

## Streamlit Dashboard Features

### Page 1: Single Prediction 📄
- Interactive form for transaction details
- Real-time fraud classification
- Risk level indicator (🟢 Low / 🟠 Medium / 🔴 High)
- Transaction analysis breakdown

### Page 2: Batch Analysis 📊
- CSV file upload
- Bulk prediction processing
- Results summary (fraud count, rate, confidence)
- Download predictions as CSV
- Progress tracking

### Page 3: Model Info 📈
- Model performance metrics (Recall, Precision, ROC-AUC)
- Model explanation & feature descriptions
- Risk level thresholds
- Training date & performance summary

## Troubleshooting

### API Won't Start
```
ImportError: email-validator version >= 2.0 required
```
**Fix:**
```bash
pip install -U email-validator
```

### Connection Refused
```
requests.exceptions.ConnectionError: Failed to establish connection
```
**Fix:** Ensure FastAPI is running:
```bash
python src/api/fastapi_app.py
# Should output: "Uvicorn running on http://127.0.0.1:8000"
```

### Docker Network Issues
```
ERR cannot access service from another container
```
**Fix:** Ensure services use container networking:
```bash
# Use service name (not localhost) in docker-compose
API_URL = "http://fastapi_backend:8000"
```

### Model File Not Found
```
FileNotFoundError: models/trained_models/logistic_regression.pkl
```
**Fix:** Run Phase 5 (Model Training) to generate models:
```bash
python src/model_training/train_models.py
```

## Performance Metrics

**FastAPI Backend:**
- Response time: ~50-100ms per prediction
- Throughput: ~10-20 predictions/second (single CPU)
- Memory: ~300MB with model loaded

**Streamlit Dashboard:**
- Load time: ~2-3 seconds
- Single prediction UI response: <100ms
- Batch processing: Depends on CSV size (~100 rows/second)

## Production Recommendations

1. **HTTPS:** Use reverse proxy (nginx) with SSL
2. **Authentication:** Add API key validation to `/predict` endpoint
3. **Rate Limiting:** Implement rate limiter to prevent abuse
4. **Monitoring:** Add Prometheus metrics & health checks
5. **Logging:** Centralize logs to ELK stack or CloudWatch
6. **Load Balancing:** Run multiple API replicas behind load balancer
7. **Database:** Connect to production PostgreSQL for transaction logging
8. **Model Versioning:** Implement model selector (MLflow Registry integration)
9. **Caching:** Add Redis for batch prediction results
10. **Documentation:** Generate API docs with Swagger/OpenAPI

## Files Modified in Phase 8

- ✅ `src/api/fastapi_app.py` - REST API with 4 endpoints
- ✅ `src/ui/streamlit_app.py` - 3-page interactive dashboard
- ✅ `Dockerfile.api` - FastAPI container image
- ✅ `Dockerfile.ui` - Streamlit container image
- ✅ `docker-compose.yml` - Multi-container orchestration
- ✅ `DEPLOYMENT.md` - This guide

## Next Steps

1. **Test locally** (FastAPI + Streamlit)
2. **Build Docker images** (`docker-compose build`)
3. **Deploy to production** (cloud platform of choice)
4. **Monitor performance** (API response times, error rates)
5. **Iterate on features** (add authentication, logging, monitoring)
