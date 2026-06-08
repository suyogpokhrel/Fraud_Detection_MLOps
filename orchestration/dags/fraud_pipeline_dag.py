"""
Fraud Detection MLOps Pipeline DAG
Orchestrates: Ingestion → Preprocessing → Feature Engineering →
              Training → MLflow Logging → Drift Monitoring
Schedule: Weekly (every Monday at midnight)
"""

from datetime import datetime, timedelta
from pathlib import Path
import sys
from airflow import DAG
from airflow.operators.python import PythonOperator

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

from src.ingestion.prepare_dataset import prepare_dataset as prepare_dataset_main
from src.ingestion.load_to_mariadb import load_processed_csv
from src.preprocessing.preprocess import PreprocessingPipeline
from src.feature_engineering.engineer_features import FeatureEngineer
from src.model_training.train_models import ModelTrainer
from src.mlflow_tracking.mlflow_integration import MLflowExperimentTracker
from src.monitoring.drift_monitor import run_drift_report

default_args = {
    'owner': 'mlops',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
    'email_on_failure': False,
}

def task_prepare_dataset():
    print("[DAG] Starting: Prepare Dataset")
    prepare_dataset_main(repo_root)
    print("[DAG] Complete: Prepare Dataset")

def task_load_to_mariadb():
    print("[DAG] Starting: Load to MariaDB")
    load_processed_csv(repo_root)
    print("[DAG] Complete: Load to MariaDB")

def task_preprocess():
    print("[DAG] Starting: Preprocessing")
    pipeline = PreprocessingPipeline()
    pipeline.run(output_dir=repo_root / "data" / "preprocessed")
    print("[DAG] Complete: Preprocessing")

def task_feature_engineering():
    print("[DAG] Starting: Feature Engineering")
    engineer = FeatureEngineer()
    engineer.run(
        input_dir=repo_root / "data" / "preprocessed",
        output_dir=repo_root / "data" / "engineered_features"
    )
    print("[DAG] Complete: Feature Engineering")

def task_train_models():
    print("[DAG] Starting: Model Training")
    trainer = ModelTrainer()
    trainer.run()
    print("[DAG] Complete: Model Training")

def task_mlflow_log():
    print("[DAG] Starting: MLflow Integration")
    tracker = MLflowExperimentTracker()
    tracker.run(input_dir=repo_root / "models" / "trained_models")
    print("[DAG] Complete: MLflow Integration")

def task_drift_monitoring():
    print("[DAG] Starting: Drift Monitoring")
    report_path = run_drift_report()
    print(f"[DAG] Complete: Drift report saved to {report_path}")

with DAG(
    dag_id='fraud_detection_pipeline',
    default_args=default_args,
    description='Fraud Detection MLOps Pipeline with weekly retraining and drift monitoring',
    schedule_interval='@weekly',
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=['mlops', 'fraud-detection'],
) as dag:

    t1_prepare = PythonOperator(
        task_id='prepare_dataset',
        python_callable=task_prepare_dataset,
    )
    t2_load = PythonOperator(
        task_id='load_to_mariadb',
        python_callable=task_load_to_mariadb,
    )
    t3_preprocess = PythonOperator(
        task_id='preprocess',
        python_callable=task_preprocess,
    )
    t4_features = PythonOperator(
        task_id='feature_engineering',
        python_callable=task_feature_engineering,
    )
    t5_train = PythonOperator(
        task_id='train_models',
        python_callable=task_train_models,
    )
    t6_mlflow = PythonOperator(
        task_id='mlflow_log',
        python_callable=task_mlflow_log,
    )
    t7_drift = PythonOperator(
        task_id='drift_monitoring',
        python_callable=task_drift_monitoring,
    )

    # Pipeline order
    t1_prepare >> t2_load >> t3_preprocess >> t4_features >> t5_train >> t6_mlflow >> t7_drift
