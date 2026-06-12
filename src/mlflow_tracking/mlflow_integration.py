
from pathlib import Path
import sys
import json
import joblib
import warnings
warnings.filterwarnings('ignore')
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))
from src.cache.redis_client import load_json

class MLflowTracker:
    def __init__(self):
        self.models_dir  = repo_root / "models" / "trained_models"
        self.tracking_uri = f"sqlite:///{repo_root}/mlflow.db"
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment("fraud_detection")

    def load_metrics(self):
        metrics = load_json("model_metrics")
        if metrics is None:
            metrics_path = self.models_dir / "metrics.json"
            metrics = json.loads(metrics_path.read_text())
            print("  Loaded metrics from file (Redis miss)")
        else:
            print("  Loaded metrics from Redis")
        return metrics

    def load_meta(self):
        meta_path = self.models_dir / "metadata.json"
        if meta_path.exists():
            return json.loads(meta_path.read_text())
        # fallback to old txt format
        meta_path = self.models_dir / "metadata.txt"
        meta = {}
        for line in meta_path.read_text().splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip()
        return meta

    def log_model(self, name, model, metrics):
        with mlflow.start_run(run_name=name):
            # log metrics
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(k.lower().replace("-", "_"), v)

            # log params
            mlflow.log_param("model_type", name)
            mlflow.log_param("dataset", "PaySim_88213_rows")
            mlflow.log_param("features", "transfer_cashout_only")

            # log model with correct flavor
            if "XGBoost" in name:
                mlflow.xgboost.log_model(model, artifact_path="model",
                                         registered_model_name=None)
            else:
                mlflow.sklearn.log_model(model, artifact_path="model",
                                         registered_model_name=None)

            run_id = mlflow.active_run().info.run_id
            print(f"  Logged {name} — run_id: {run_id[:8]}...")
            return run_id

    def register_best(self, best_name, best_run_id):
        model_uri = f"runs:/{best_run_id}/model"
        reg = mlflow.register_model(model_uri, "fraud_detection_best_model")
        print(f"  Registered '{best_name}' as fraud_detection_best_model v{reg.version}")

    def run(self, input_dir=None):
        print("=" * 55)
        print("PHASE 6: MLFLOW TRACKING")
        print("=" * 55)

        metrics = self.load_metrics()
        meta    = self.load_meta()
        best_name = meta.get("best_model", "XGBoost")

        model_files = {
            "Logistic Regression": "logistic_regression.pkl",
            "Random Forest":       "random_forest.pkl",
            "XGBoost":             "xgboost.pkl",
        }

        best_run_id = None
        for name, fname in model_files.items():
            fpath = self.models_dir / fname
            if not fpath.exists():
                print(f"  Skipping {name} — file not found")
                continue
            model = joblib.load(fpath)
            run_id = self.log_model(name, model, metrics.get(name, {}))
            if name == best_name:
                best_run_id = run_id

        if best_run_id:
            self.register_best(best_name, best_run_id)

        print("\n✓ MLFLOW TRACKING COMPLETE")

if __name__ == "__main__":
    tracker = MLflowTracker()
    tracker.run()
