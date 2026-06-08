"""
Phase 7: Airflow DAG for Fraud Detection Pipeline Automation
Orchestrates: Ingestion → Preprocessing → Feature Engineering → Training → MLflow Logging
Uses: PythonOperators (direct function calls, not bash commands)
"""

from datetime import datetime, timedelta
from pathlib import Path
import sys
from airflow import DAG
from airflow.operators.python import PythonOperator

# Set up repository path
repo_root = Path('/home/suyog/Desktop/A_space/MLOPS/Final_Project/Fraud_Detection_MLOps')
sys.path.insert(0, str(repo_root))

# Import pipeline modules
from src.ingestion.prepare_dataset import prepare_dataset as prepare_dataset_main
from src.ingestion.load_to_mariadb import load_processed_csv
from src.preprocessing.preprocess import PreprocessingPipeline
from src.feature_engineering.engineer_features import FeatureEngineer
from src.model_training.train_models import ModelTrainer
from src.mlflow_tracking.mlflow_integration import MLflowExperimentTracker

# DAG default args
default_args = {
    'owner': 'mlops',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
}

# Task wrapper functions
def task_prepare_dataset():
    """Phase 1: Prepare raw dataset for ingestion"""
    print("[DAG] Starting: Prepare Dataset")
    prepare_dataset_main(repo_root)
    print("[DAG] Complete: Prepare Dataset")

def task_load_to_mariadb():
    """Phase 2: Load preprocessed data to MariaDB"""
    print("[DAG] Starting: Load to MariaDB")
    load_processed_csv(repo_root)
    print("[DAG] Complete: Load to MariaDB")

def task_preprocess():
    """Phase 3: Preprocess data (scaling, encoding)"""
    print("[DAG] Starting: Preprocessing")
    pipeline = PreprocessingPipeline()
    pipeline.run(output_dir=repo_root / "data" / "preprocessed")
    print("[DAG] Complete: Preprocessing")

def task_feature_engineering():
    """Phase 4: Engineer domain-specific features"""
    print("[DAG] Starting: Feature Engineering")
    engineer = FeatureEngineer()
    engineer.run(
        input_dir=repo_root / "data" / "preprocessed",
        output_dir=repo_root / "data" / "engineered_features"
    )
    print("[DAG] Complete: Feature Engineering")

def task_train_models():
    """Phase 5: Train and evaluate models"""
    print("[DAG] Starting: Model Training")
    trainer = ModelTrainer()
    trainer.run()
    print("[DAG] Complete: Model Training")

def task_mlflow_log():
    """Phase 6: Log models and metrics to MLflow"""
    print("[DAG] Starting: MLflow Integration")
    tracker = MLflowExperimentTracker()
    tracker.run(input_dir=repo_root / "models" / "trained_models")
    print("[DAG] Complete: MLflow Integration")

# Create DAG
with DAG(
    dag_id='fraud_detection_pipeline',
    default_args=default_args,
    description='Fraud Detection MLOps Pipeline: Ingestion → Preprocessing → Features → Training → MLflow',
    schedule_interval=None,  # Manual trigger
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=['mlops', 'fraud-detection'],
) as dag:

    # Create PythonOperator tasks
    t1_prepare = PythonOperator(
        task_id='prepare_dataset',
        python_callable=task_prepare_dataset,
        trigger_rule='all_success',
    )

    t2_load = PythonOperator(
        task_id='load_to_mariadb',
        python_callable=task_load_to_mariadb,
        trigger_rule='all_success',
    )

    t3_preprocess = PythonOperator(
        task_id='preprocess',
        python_callable=task_preprocess,
        trigger_rule='all_success',
    )

    t4_features = PythonOperator(
        task_id='feature_engineering',
        python_callable=task_feature_engineering,
        trigger_rule='all_success',
    )

    t5_train = PythonOperator(
        task_id='train_models',
        python_callable=task_train_models,
        trigger_rule='all_success',
    )

    t6_mlflow = PythonOperator(
        task_id='mlflow_log',
        python_callable=task_mlflow_log,
        trigger_rule='all_success',
    )

    # DAG ordering (pipeline dependencies)
    t1_prepare >> t2_load >> t3_preprocess >> t4_features >> t5_train >> t6_mlflow

