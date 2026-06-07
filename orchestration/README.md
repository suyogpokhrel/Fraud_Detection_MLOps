# Airflow DAG for Fraud Detection Pipeline

## Overview
Phase 7: Local Airflow orchestration using PythonOperators to run the fraud detection pipeline:
1. Prepare Dataset
2. Load to MariaDB
3. Preprocess
4. Feature Engineering
5. Train Models
6. MLflow Logging

## Files
- `orchestration/dags/fraud_pipeline_dag.py` — Main DAG with PythonOperators
- `orchestration/README.md` — This file

## Local Setup & Execution

### Step 1: Install Airflow

Install Airflow in your `fraud_mlops` conda environment:

```bash
conda activate fraud_mlops
pip install apache-airflow==2.5.1
```

### Step 2: Initialize Airflow Database

```bash
export AIRFLOW_HOME="$PWD/orchestration"
airflow db init
```

This creates `orchestration/airflow.db` (SQLite) for local testing.

### Step 3: Create Airflow Admin User

```bash
airflow users create \
  --username admin \
  --firstname Admin \
  --lastname User \
  --role Admin \
  --email admin@example.com \
  --password admin
```

### Step 4: Start Airflow Webserver & Scheduler

Open two terminal windows:

**Terminal 1 (Webserver):**
```bash
export AIRFLOW_HOME="$PWD/orchestration"
airflow webserver --port 8080
```

**Terminal 2 (Scheduler):**
```bash
export AIRFLOW_HOME="$PWD/orchestration"
airflow scheduler
```

### Step 5: Access Airflow UI

Open browser and go to: **http://127.0.0.1:8080**

Login with:
- Username: `admin`
- Password: `admin`

### Step 6: Trigger DAG

From the Airflow UI:
1. Find `fraud_detection_pipeline` in the DAG list
2. Click the DAG name to view details
3. Click **Trigger DAG** (green button)
4. Monitor task execution in real-time

**Or from CLI:**
```bash
airflow dags trigger fraud_detection_pipeline
```

## DAG Structure

```
prepare_dataset →
load_to_mariadb →
preprocess →
feature_engineering →
train_models →
mlflow_log
```

Each task is a PythonOperator that calls the corresponding pipeline module directly.

## Notes

- All data/models are saved to `data/` and `models/` directories in the repo root
- MLflow artifacts go to `mlruns/`
- For production, use a real Postgres backend (not SQLite) and distributed executor (Celery/Kubernetes)
- Environment variables (DB_HOST, DB_USER, DB_PASSWORD) sourced from shell
