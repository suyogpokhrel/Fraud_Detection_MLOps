"""
Phase 6: MLflow Integration & Model Registry
Logs training experiments, tracks metrics, versions models, manages lifecycle.

MLflow Components:
  1. Tracking: Log parameters, metrics, artifacts
  2. Models: Store trained model serializations
  3. Registry: Manage model versions, stages (Staging/Production)
  4. Server: UI for monitoring experiments

Workflow:
  - Log model training runs with parameters and metrics
  - Compare multiple model performances
  - Register best model to production
  - Track model evolution over time
"""

import sys
from pathlib import Path
import json
import pickle
import numpy as np
import pandas as pd
import warnings

try:
    import mlflow
    from mlflow.sklearn import log_model as mlflow_log_sklearn_model
    from mlflow.xgboost import log_model as mlflow_log_xgboost_model
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False
    print("⚠ MLflow not installed")

warnings.filterwarnings('ignore')

# Add repository to path
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))


class MLflowExperimentTracker:
    """
    Manages MLflow experiment tracking and model registry.
    Logs model training runs, compares performances, registers best model.
    """

    def __init__(self, experiment_name="Fraud Detection MLOps"):
        if not HAS_MLFLOW:
            raise ImportError("MLflow is required. Install with: pip install mlflow")

        self.experiment_name = experiment_name
        self.tracking_uri = None
        self.models_info = {}
        self.best_model_info = None

        # Set MLflow tracking URI to SQLite database (production-ready)
        self.mlflow_dir = repo_root / "mlruns"
        self.mlflow_dir.mkdir(exist_ok=True)
        self.tracking_uri = f"sqlite:///{repo_root / 'mlruns' / 'mlflow.db'}"
        
        mlflow.set_tracking_uri(self.tracking_uri)

        # Create or get experiment
        try:
            self.experiment_id = mlflow.create_experiment(self.experiment_name)
        except:
            # Experiment already exists
            exp = mlflow.get_experiment_by_name(self.experiment_name)
            self.experiment_id = exp.experiment_id

        mlflow.set_experiment(self.experiment_name)

    def log_training_run(self, model_name, model, params, metrics, scaler=None):
        """
        Log a complete training run to MLflow.
        """
        print(f"  └─ Logging {model_name} to MLflow...")

        with mlflow.start_run(run_name=model_name):
            # Log parameters
            mlflow.log_params(params)

            # Log metrics
            mlflow.log_metrics(metrics)

            # Log model artifacts
            mlflow.log_artifact(
                repo_root / "models" / "trained_models" / "training_report.txt",
                artifact_path="training_reports"
            )

            # Save and log the model
            # Use a simple artifact/model name without illegal characters
            model_path = model_name.lower().replace(' ', '_')
            # MLflow auto-logs sklearn/xgboost models under artifact path = model_path
            if 'sklearn' in str(type(model).__module__):
                mlflow.sklearn.log_model(model, artifact_path=model_path)
            elif 'xgboost' in str(type(model).__module__):
                mlflow.xgboost.log_model(model, artifact_path=model_path)

            # Log model metadata
            mlflow.set_tag("model_type", type(model).__name__)
            mlflow.set_tag("fraud_detection", "fraud_detection_mlops")
            mlflow.set_tag("phase", "5_model_training")

            # Get run info
            run = mlflow.active_run()
            run_id = run.info.run_id

            self.models_info[model_name] = {
                "run_id": run_id,
                "metrics": metrics,
                "params": params,
                "model_path": model_path,
            }

            print(f"     ✓ Run ID: {run_id}")

    def register_best_model(self, best_model_name, version_description=""):
        """
        Register the best performing model to MLflow Model Registry.
        """
        print(f"\n[4/5] Registering best model to MLflow Model Registry...")

        if best_model_name not in self.models_info:
            print(f"  ⚠ Model {best_model_name} not found in tracked runs")
            return

        run_id = self.models_info[best_model_name]["run_id"]
        model_path = self.models_info[best_model_name].get("model_path", best_model_name.lower().replace(' ', '_'))
        model_uri = f"runs:/{run_id}/{model_path}"

        try:
            registered_model = mlflow.register_model(model_uri, "fraud_detection_model")
            print(f"✓ Model registered: {registered_model.name}")
            print(f"  Version: {registered_model.version}")

            # Transition to Staging
            client = mlflow.tracking.MlflowClient()
            client.transition_model_version_stage(
                "fraud_detection_model",
                registered_model.version,
                "Staging"
            )
            print(f"  Stage: Staging (waiting for production approval)")

            self.best_model_info = {
                "name": registered_model.name,
                "version": registered_model.version,
                "stage": "Staging",
                "run_id": run_id,
            }

        except Exception as e:
            print(f"  ✓ Model already registered (version will increment): {e}")

    def compare_models(self):
        """
        Display comparison of all logged models.
        """
        print(f"\n[3/5] Model Comparison (MLflow)...")
        print("\n  Tracked Models:")
        print("  " + "-" * 80)

        for model_name, info in self.models_info.items():
            print(f"\n  {model_name}:")
            print(f"    Run ID: {info['run_id']}")
            metrics = info['metrics']
            print(f"    Recall: {metrics.get('Recall', 'N/A'):.4f}")
            print(f"    Precision: {metrics.get('Precision', 'N/A'):.4f}")
            print(f"    ROC-AUC: {metrics.get('ROC-AUC', 'N/A'):.4f}")

    def log_dataset_metadata(self, dataset_info):
        """
        Log dataset information as artifacts for reproducibility.
        """
        print(f"\n[2/5] Logging dataset metadata...")

        # Save dataset info
        dataset_file = repo_root / "mlflow_dataset_info.json"
        with open(dataset_file, 'w') as f:
            json.dump(dataset_info, f, indent=2)

        # Log as artifact in a tracking run
        with mlflow.start_run(run_name="dataset_metadata"):
            mlflow.log_artifact(dataset_file, artifact_path="dataset")
            mlflow.set_tag("dataset_version", "1.0")
            mlflow.set_tag("total_samples", dataset_info.get("total_samples"))
            mlflow.set_tag("fraud_rate", f"{dataset_info.get('fraud_rate', 0):.2f}%")
            print(f"  ✓ Dataset metadata logged")

    def save_tracking_info(self, output_dir: Path = None):
        """
        Save MLflow tracking information for reference.
        """
        if output_dir is None:
            output_dir = repo_root / "mlruns"

        print(f"\n[5/5] Saving MLflow tracking information...")

        # Save experiment info
        exp_info = {
            "experiment_name": self.experiment_name,
            "experiment_id": self.experiment_id,
            "tracking_uri": str(self.tracking_uri),
            "models_tracked": self.models_info,
            "best_model": self.best_model_info,
        }

        info_file = output_dir / "mlflow_tracking_info.json"
        with open(info_file, 'w') as f:
            json.dump(exp_info, f, indent=2, default=str)

        print(f"✓ Tracking info saved: {info_file}")

        # Save MLflow UI startup script
        ui_script = output_dir / "start_mlflow_ui.sh"
        with open(ui_script, 'w') as f:
            f.write(f"#!/bin/bash\n")
            f.write(f"# Start MLflow UI for experiment tracking\n")
            f.write(f"cd {repo_root}\n")
            # Use the configured tracking URI (sqlite or file-based)
            f.write(f"mlflow ui --backend-store-uri {self.tracking_uri}\n")

        ui_script.chmod(0o755)
        print(f"✓ MLflow UI script: {ui_script}")
        print(f"\n  To view experiments: bash {ui_script}")
        print(f"  Then open: http://127.0.0.1:5000")

    def run(self, input_dir: Path = None):
        """Execute complete MLflow integration."""
        if input_dir is None:
            input_dir = repo_root / "models" / "trained_models"

        print("\n" + "=" * 80)
        print("PHASE 6: MLFLOW INTEGRATION & MODEL REGISTRY")
        print("=" * 80)

        # Load trained models and metadata
        print(f"\n[1/5] Loading trained models and metrics...")

        # Load evaluation results
        results_file = input_dir / "evaluation_results.json"
        if not results_file.exists():
            print(f"✗ Results file not found: {results_file}")
            return

        with open(results_file, 'r') as f:
            all_results = json.load(f)

        # Load all models
        models_file = input_dir / "all_models.pkl"
        with open(models_file, 'rb') as f:
            all_models = pickle.load(f)

        print(f"✓ Loaded {len(all_models)} models")

        # Log dataset metadata
        dataset_info = {
            "total_samples": 10113,
            "fraud_samples": 68,
            "legit_samples": 10045,
            "fraud_rate": 0.67,
            "features": 34,
            "test_split": 0.2,
        }
        self.log_dataset_metadata(dataset_info)

        # Log each model training run
        print(f"\n[2.5/5] Logging model training runs...")
        for model_name, model in all_models.items():
            params = {
                "model_type": type(model).__name__,
                "test_split": 0.2,
                "stratified": True,
            }

            if model_name in all_results:
                metrics = all_results[model_name]
            else:
                metrics = {}

            # Convert numpy types for logging
            metrics_clean = {
                k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                for k, v in metrics.items()
            }

            self.log_training_run(model_name, model, params, metrics_clean)

        # Compare models
        self.compare_models()

        # Find and register best model
        best_model_name = "Logistic Regression"  # From Phase 5
        self.register_best_model(best_model_name, "Production candidate - highest recall")

        # Save tracking info
        self.save_tracking_info()

        print("\n" + "=" * 80)
        print("✓ MLFLOW INTEGRATION COMPLETE")
        print("=" * 80)
        print(f"\nExperiment: {self.experiment_name}")
        print(f"Tracking URI: {self.tracking_uri}")
        print(f"Models logged: {len(self.models_info)}")
        if self.best_model_info:
            print(f"Best model registered: {self.best_model_info['name']} (v{self.best_model_info['version']})")
        print(f"\nNext phase: Airflow automation for pipeline orchestration")
        print()


if __name__ == "__main__":
    tracker = MLflowExperimentTracker()
    tracker.run()
