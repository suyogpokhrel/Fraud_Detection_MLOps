"""
Phase 5: Fraud Model Training (STABLE + BEST MODEL SELECTION)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import json
import warnings

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, average_precision_score
)

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

warnings.filterwarnings("ignore")

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))


class ModelTrainer:

    def __init__(self):
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.models = {}
        self.results = {}

        self.best_model_name = None
        self.best_model = None

        self.scaler = StandardScaler()

    # =========================
    # LOAD DATA
    # =========================
    def load_data(self):

        print("\n[1/6] Loading data...")

        X = pd.read_csv(repo_root / "data" / "engineered_features" / "X_engineered.csv").values
        y = pd.read_csv(repo_root / "data" / "engineered_features" / "y_engineered.csv").squeeze().values

        print(f"✓ Shape: {X.shape}")
        print(f"✓ Fraud rate: {100*y.mean():.2f}%")

        return X, y

    # =========================
    # SPLIT + SMOTE
    # =========================
    def split_data(self, X, y):

        print("\n[2/6] Train-test split...")

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=0.2,
            stratify=y,
            random_state=42
        )

        print(f"✓ Train: {len(self.X_train)} | Test: {len(self.X_test)}")

        print("\n[2.1] Applying SMOTE...")

        smote = SMOTE(
            sampling_strategy=0.1,
            random_state=42
        )

        self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)

        self.X_train = self.X_train.astype(np.float32)
        self.X_test = self.X_test.astype(np.float32)

        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

    # =========================
    # TUNING MODELS
    # =========================
    def tune_models(self):

        print("\n[3/6] Hyperparameter Tuning...")

        self.models = {}

        # Logistic Regression
        log_search = RandomizedSearchCV(
            LogisticRegression(max_iter=1000),
            {"C": [0.1, 1]},
            n_iter=2,
            scoring="recall",
            cv=2,
            n_jobs=1
        )
        log_search.fit(self.X_train, self.y_train)
        self.models["Logistic Regression"] = log_search.best_estimator_

        print("✓ LR:", log_search.best_params_)

        # Random Forest
        rf_search = RandomizedSearchCV(
            RandomForestClassifier(n_jobs=1),
            {
                "n_estimators": [100, 150],
                "max_depth": [10, 15]
            },
            n_iter=3,
            scoring="recall",
            cv=2,
            n_jobs=1
        )
        rf_search.fit(self.X_train, self.y_train)
        self.models["Random Forest"] = rf_search.best_estimator_

        print("✓ RF:", rf_search.best_params_)

        # XGBoost
        if HAS_XGBOOST:

            xgb_search = RandomizedSearchCV(
                XGBClassifier(
                    n_jobs=1,
                    tree_method="hist"
                ),
                {
                    "n_estimators": [100],
                    "max_depth": [3, 6],
                    "learning_rate": [0.1]
                },
                n_iter=2,
                scoring="recall",
                cv=2,
                n_jobs=1
            )

            xgb_search.fit(self.X_train, self.y_train)
            self.models["XGBoost"] = xgb_search.best_estimator_

            print("✓ XGBoost:", xgb_search.best_params_)

    # =========================
    # EVALUATION + THRESHOLD
    # =========================
    def _evaluate(self, name, model):

        y_proba = model.predict_proba(self.X_test)[:, 1]

        best_metrics = None
        best_threshold = 0.5

        for threshold in np.arange(0.1, 0.9, 0.01):

            y_pred = (y_proba >= threshold).astype(int)

            p = precision_score(self.y_test, y_pred, zero_division=0)
            r = recall_score(self.y_test, y_pred, zero_division=0)

            if p >= 0.50 and r >= 0.70:
                best_threshold = threshold
                best_metrics = self._metrics(y_pred, y_proba)
                break

        if best_metrics is None:

            best_f1 = -1

            for threshold in np.arange(0.1, 0.9, 0.01):

                y_pred = (y_proba >= threshold).astype(int)
                f1 = f1_score(self.y_test, y_pred, zero_division=0)

                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
                    best_metrics = self._metrics(y_pred, y_proba)

        self.results[name] = best_metrics

        print(f"\n📊 {name}")
        print(f"  Precision: {best_metrics['Precision']:.4f}")
        print(f"  Recall   : {best_metrics['Recall']:.4f}")
        print(f"  F1       : {best_metrics['F1']:.4f}")
        print(f"  ROC-AUC  : {best_metrics['ROC-AUC']:.4f}")

    # =========================
    # METRICS
    # =========================
    def _metrics(self, y_pred, y_proba):

        tn, fp, fn, tp = confusion_matrix(self.y_test, y_pred).ravel()

        return {
            "Precision": precision_score(self.y_test, y_pred, zero_division=0),
            "Recall": recall_score(self.y_test, y_pred, zero_division=0),
            "F1": f1_score(self.y_test, y_pred, zero_division=0),
            "Accuracy": accuracy_score(self.y_test, y_pred),
            "ROC-AUC": roc_auc_score(self.y_test, y_proba),
            "PR-AUC": average_precision_score(self.y_test, y_proba),
            "TP": int(tp),
            "FP": int(fp),
            "FN": int(fn),
            "TN": int(tn)
        }

    # =========================
    # BEST MODEL SELECTION (NEW)
    # =========================
    def select_best_model(self):

        print("\n[4/6] Selecting best model...")

        best_score = -1

        for name, m in self.results.items():

            score = (0.5 * m["Precision"]) + (0.5 * m["Recall"])

            print(f"\n{name}")
            print(f"  Precision: {m['Precision']:.4f}")
            print(f"  Recall   : {m['Recall']:.4f}")
            print(f"  Score    : {score:.4f}")

            if score > best_score:
                best_score = score
                self.best_model_name = name

        self.best_model = self.models[self.best_model_name]

        print(f"\n🏆 BEST MODEL: {self.best_model_name}")

    # =========================
    # SAVE BEST MODEL
    # =========================
    def save_model(self):

        output_dir = repo_root / "models" / "trained_models"
        output_dir.mkdir(parents=True, exist_ok=True)

        print("\n[5/6] Saving best model...")

        with open(output_dir / "best_model.pkl", "wb") as f:
            pickle.dump(self.best_model, f)

        with open(output_dir / "all_models.pkl", "wb") as f:
            pickle.dump(self.models, f)

        with open(output_dir / "metrics.json", "w") as f:
            json.dump(self.results, f, indent=2)

        print("✓ Saved models + metrics")

    # =========================
    # RUN PIPELINE
    # =========================
    def run(self):

        print("\n" + "=" * 60)
        print("PHASE 5: FINAL FRAUD PIPELINE")
        print("=" * 60)

        X, y = self.load_data()
        self.split_data(X, y)

        self.tune_models()

        for name, model in self.models.items():
            self._evaluate(name, model)

        self.select_best_model()
        self.save_model()

        print("\n✓ TRAINING COMPLETE")


if __name__ == "__main__":
    ModelTrainer().run()