
from pathlib import Path
import sys
import json
import joblib
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             accuracy_score, roc_auc_score,
                             average_precision_score, confusion_matrix)
import xgboost as xgb

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))
from src.cache.redis_client import load_dataframe, save_dataframe, save_json, load_json

class ModelTrainer:
    def __init__(self):
        self.output_dir = repo_root / "models" / "trained_models"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}

    def load_data(self):
        print("Loading engineered features...")
        X = load_dataframe("engineered_X")
        y = load_dataframe("engineered_y")
        if X is None:
            X = pd.read_csv(repo_root / "data" / "engineered_features" / "X_engineered.csv")
            y = pd.read_csv(repo_root / "data" / "engineered_features" / "y_engineered.csv").squeeze()
            print("  Loaded from CSV (Redis miss)")
        else:
            y = y.squeeze()
            print("  Loaded from Redis")
        print(f"  X shape: {X.shape}, fraud rate: {y.mean():.4f}")
        return X, y

    def evaluate(self, model, X_test, y_test, name):
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        metrics = {
            "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "Recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
            "F1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
            "Accuracy":  round(accuracy_score(y_test, y_pred), 4),
            "ROC-AUC":   round(roc_auc_score(y_test, y_proba), 4),
            "PR-AUC":    round(average_precision_score(y_test, y_proba), 4),
            "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn)
        }
        print(f"  {name}: Precision={metrics['Precision']:.4f}  "
              f"Recall={metrics['Recall']:.4f}  "
              f"F1={metrics['F1']:.4f}  ROC-AUC={metrics['ROC-AUC']:.4f}")
        return metrics

    def train(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
        print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
        print(f"  Fraud in train: {y_train.sum()}, test: {y_test.sum()}")

        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        models = {
            "Logistic Regression": LogisticRegression(
                class_weight='balanced', max_iter=1000, random_state=42),
            "Random Forest": RandomForestClassifier(
                n_estimators=100, class_weight='balanced',
                max_depth=10, random_state=42, n_jobs=-1),
            "XGBoost": xgb.XGBClassifier(
                scale_pos_weight=scale_pos_weight,
                n_estimators=200, max_depth=6,
                learning_rate=0.05, subsample=0.8,
                colsample_bytree=0.8, reg_alpha=0.1,
                reg_lambda=1.0, random_state=42,
                eval_metric='aucpr', verbosity=0)
        }

        print("\nTraining models with 5-fold cross-validation...")
        trained = {}
        for name, model in models.items():
            print(f"\n  [{name}]")
            cv_scores = cross_val_score(model, X_train, y_train,
                                        cv=skf, scoring='recall', n_jobs=-1)
            print(f"  CV Recall: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
            model.fit(X_train, y_train)
            metrics = self.evaluate(model, X_test, y_test, name)
            self.results[name] = metrics
            trained[name] = model

        return trained, X_test, y_test

    def select_best(self, trained):
        # Best = highest F1 score (balances precision and recall)
        best_name = max(self.results, key=lambda n: self.results[n]["F1"])
        print(f"\n  Best model: {best_name} (F1={self.results[best_name]['F1']:.4f})")
        return best_name, trained[best_name]

    def save(self, trained, best_name, best_model, X, y):
        for name, model in trained.items():
            fname = name.lower().replace(" ", "_") + ".pkl"
            joblib.dump(model, self.output_dir / fname)
        joblib.dump(best_model, self.output_dir / "best_model.pkl")

        scaler_src = repo_root / "data" / "preprocessed" / "scaler.joblib"
        if scaler_src.exists():
            import shutil
            shutil.copy(scaler_src, self.output_dir / "scaler.pkl")

        feature_names = list(X.columns)
        (self.output_dir / "feature_names.json").write_text(json.dumps(feature_names))

        metrics_path = self.output_dir / "metrics.json"
        metrics_path.write_text(json.dumps(self.results, indent=2))

        fraud_rate = float(y.mean())
        meta = {
            "best_model": best_name,
            "total_models_trained": len(trained),
            "feature_count": len(feature_names),
            "fraud_rate_full": round(fraud_rate, 4),
            "best_recall": self.results[best_name]["Recall"],
            "best_f1": self.results[best_name]["F1"],
            "best_roc_auc": self.results[best_name]["ROC-AUC"],
        }
        (self.output_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

        save_json(self.results, "model_metrics")
        print("\n  All models saved.")

    def run(self):
        print("=" * 55)
        print("PHASE 5: MODEL TRAINING")
        print("=" * 55)
        X, y = self.load_data()
        trained, X_test, y_test = self.train(X, y)
        best_name, best_model = self.select_best(trained)
        self.save(trained, best_name, best_model, X, y)
        print("\n✓ MODEL TRAINING COMPLETE")
        print(f"  Best: {best_name}")
        print(f"  Recall={self.results[best_name]['Recall']:.4f}  "
              f"F1={self.results[best_name]['F1']:.4f}  "
              f"ROC-AUC={self.results[best_name]['ROC-AUC']:.4f}")
        return best_name, self.results

if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.run()
