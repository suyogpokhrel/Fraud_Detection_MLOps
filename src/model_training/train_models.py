"""
Phase 5: Model Training & Evaluation Pipeline
Trains multiple models, evaluates with fraud-specific metrics, selects best model.

Models:
  1. Logistic Regression (baseline, interpretable)
  2. Random Forest (robust, handles non-linearity)
  3. XGBoost (state-of-the-art, powerful)

Evaluation Metrics (Fraud-Specific):
  - Recall: Catch as many frauds as possible (minimize false negatives)
  - Precision: Avoid false alarms (minimize false positives)
  - F1: Balanced harmonic mean
  - ROC-AUC: Performance across all thresholds
  - PR-AUC: Precision-Recall curve (better for imbalanced data)

Notes:
  - Data is highly imbalanced (0.67% fraud), so:
    * Use stratified train/test split to preserve distribution
    * Use class_weight='balanced' to penalize minority misclassification
    * Use recall-focused thresholds for fraud detection
    * Prioritize Recall > Precision (catch frauds > avoid false positives)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import json
import warnings

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, auc, precision_recall_curve, roc_curve,
    confusion_matrix, classification_report, average_precision_score
)

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("⚠ XGBoost not installed, will use Random Forest + Logistic Regression")

warnings.filterwarnings('ignore')

# Add repository to path
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))


class ModelTrainer:
    """
    Train and evaluate multiple fraud detection models.
    Handles imbalanced data with stratified splits and class weighting.
    """

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

    def load_data(self, input_dir: Path = None):
        """Load engineered features from Phase 4."""
        if input_dir is None:
            input_dir = repo_root / "data" / "engineered_features"

        print("\n[1/6] Loading engineered features...")
        X_path = input_dir / "X_engineered.csv"
        y_path = input_dir / "y_engineered.csv"

        X = pd.read_csv(X_path).values  # Convert to numpy
        y = pd.read_csv(y_path).squeeze().values

        print(f"✓ Loaded: {X.shape[0]} samples × {X.shape[1]} features")
        print(f"  Class distribution: {np.bincount(y)} (0: legit, 1: fraud)")
        print(f"  Fraud rate: {100*y.mean():.2f}%")

        return X, y

    def split_data(self, X, y, test_size=0.2, random_state=42):
        """
        Stratified train/test split to preserve fraud ratio in both sets.
        """
        print("\n[2/6] Splitting data (stratified)...")

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y  # Preserve class distribution
        )

        print(f"✓ Train set: {len(self.X_train)} samples ({100*self.y_train.mean():.2f}% fraud)")
        print(f"✓ Test set:  {len(self.X_test)} samples ({100*self.y_test.mean():.2f}% fraud)")

        # Scale features
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        print(f"✓ Features scaled (StandardScaler)")

    def train_logistic_regression(self):
        """Train Logistic Regression (baseline, interpretable)."""
        print("\n[3/6] Training models...")
        print("  └─ Logistic Regression (baseline)...")

        model = LogisticRegression(
            max_iter=1000,
            class_weight='balanced',  # Handle imbalance
            random_state=42,
            n_jobs=-1
        )

        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]

        self.models['Logistic Regression'] = model
        self.results['Logistic Regression'] = self._evaluate_model(
            y_pred, y_pred_proba, 'Logistic Regression'
        )

    def train_random_forest(self):
        """Train Random Forest (robust, handles non-linearity)."""
        print("  └─ Random Forest...")

        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )

        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]

        self.models['Random Forest'] = model
        self.results['Random Forest'] = self._evaluate_model(
            y_pred, y_pred_proba, 'Random Forest'
        )

    def train_xgboost(self):
        """Train XGBoost (state-of-the-art, powerful)."""
        if not HAS_XGBOOST:
            print("  └─ XGBoost (skipped - not installed)")
            return

        print("  └─ XGBoost...")

        # Calculate scale_pos_weight for class imbalance
        neg_count = np.sum(self.y_train == 0)
        pos_count = np.sum(self.y_train == 1)
        scale_pos_weight = neg_count / pos_count

        model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1
        )

        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]

        self.models['XGBoost'] = model
        self.results['XGBoost'] = self._evaluate_model(
            y_pred, y_pred_proba, 'XGBoost'
        )

    def _evaluate_model(self, y_pred, y_pred_proba, model_name):
        """
        Comprehensive evaluation with fraud-specific metrics.
        """
        metrics = {
            'Accuracy': accuracy_score(self.y_test, y_pred),
            'Precision': precision_score(self.y_test, y_pred, zero_division=0),
            'Recall': recall_score(self.y_test, y_pred, zero_division=0),
            'F1': f1_score(self.y_test, y_pred, zero_division=0),
            'ROC-AUC': roc_auc_score(self.y_test, y_pred_proba),
            'PR-AUC': average_precision_score(self.y_test, y_pred_proba),
        }

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(self.y_test, y_pred).ravel()
        metrics['True Negatives'] = int(tn)
        metrics['False Positives'] = int(fp)
        metrics['False Negatives'] = int(fn)
        metrics['True Positives'] = int(tp)

        # Fraud-specific rates
        metrics['Fraud Catch Rate'] = metrics['Recall']  # TP / (TP + FN)
        metrics['False Alarm Rate'] = metrics['False Positives'] / (
            metrics['False Positives'] + metrics['True Negatives']
        ) if (metrics['False Positives'] + metrics['True Negatives']) > 0 else 0

        return metrics

    def select_best_model(self):
        """
        Select best model based on Recall (catch frauds) as primary metric.
        Tiebreaker: ROC-AUC (overall discrimination ability)
        """
        print("\n[4/6] Model Selection...")
        print("\n  Model Performance:")
        print("  " + "-" * 80)

        for model_name, metrics in self.results.items():
            print(f"\n  {model_name}:")
            print(f"    Recall (Fraud Catch Rate):  {metrics['Recall']:.4f}")
            print(f"    Precision:                   {metrics['Precision']:.4f}")
            print(f"    F1:                          {metrics['F1']:.4f}")
            print(f"    ROC-AUC:                     {metrics['ROC-AUC']:.4f}")
            print(f"    PR-AUC:                      {metrics['PR-AUC']:.4f}")
            print(f"    Confusion Matrix: TP={metrics['True Positives']}, FP={metrics['False Positives']}, "
                  f"FN={metrics['False Negatives']}, TN={metrics['True Negatives']}")

        # Select model with highest recall (catch frauds), tiebreak by ROC-AUC
        best_recall = -1
        best_auc = -1
        for model_name, metrics in self.results.items():
            if metrics['Recall'] > best_recall or (
                metrics['Recall'] == best_recall and metrics['ROC-AUC'] > best_auc
            ):
                best_recall = metrics['Recall']
                best_auc = metrics['ROC-AUC']
                self.best_model_name = model_name

        self.best_model = self.models[self.best_model_name]
        print(f"\n✓ Best Model: {self.best_model_name}")
        print(f"  (Selected by highest Recall={best_recall:.4f}, ROC-AUC={best_auc:.4f})")

    def save_outputs(self, output_dir: Path = None):
        """Save models, metrics, and evaluation reports."""
        if output_dir is None:
            output_dir = repo_root / "models" / "trained_models"

        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[5/6] Saving outputs to {output_dir}...")

        # Save best model
        best_model_path = output_dir / f"{self.best_model_name.lower().replace(' ', '_')}.pkl"
        with open(best_model_path, 'wb') as f:
            pickle.dump(self.best_model, f)
        print(f"  ✓ Best model: {best_model_path}")

        # Save all models
        all_models_path = output_dir / "all_models.pkl"
        with open(all_models_path, 'wb') as f:
            pickle.dump(self.models, f)
        print(f"  ✓ All models: {all_models_path}")

        # Save scaler (for inference)
        scaler_path = output_dir / "scaler.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"  ✓ Scaler: {scaler_path}")

        # Save evaluation results
        results_path = output_dir / "evaluation_results.json"
        with open(results_path, 'w') as f:
            # Convert numpy types to native Python for JSON serialization
            json_results = {}
            for model_name, metrics in self.results.items():
                json_results[model_name] = {
                    k: float(v) if isinstance(v, (np.floating, np.integer)) else int(v)
                    for k, v in metrics.items()
                }
            json.dump(json_results, f, indent=2)
        print(f"  ✓ Results: {results_path}")

        # Save detailed report
        report_path = output_dir / "training_report.txt"
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("FRAUD DETECTION MODEL TRAINING REPORT\n")
            f.write("=" * 80 + "\n\n")

            f.write("DATASET INFORMATION:\n")
            f.write(f"  Train samples: {len(self.X_train)}\n")
            f.write(f"  Test samples: {len(self.X_test)}\n")
            f.write(f"  Fraud rate (train): {100*self.y_train.mean():.2f}%\n")
            f.write(f"  Fraud rate (test): {100*self.y_test.mean():.2f}%\n\n")

            f.write("MODEL PERFORMANCE COMPARISON:\n")
            f.write("-" * 80 + "\n")
            for model_name, metrics in self.results.items():
                f.write(f"\n{model_name}:\n")
                f.write(f"  Accuracy:      {metrics['Accuracy']:.4f}\n")
                f.write(f"  Precision:     {metrics['Precision']:.4f}\n")
                f.write(f"  Recall:        {metrics['Recall']:.4f}\n")
                f.write(f"  F1:            {metrics['F1']:.4f}\n")
                f.write(f"  ROC-AUC:       {metrics['ROC-AUC']:.4f}\n")
                f.write(f"  PR-AUC:        {metrics['PR-AUC']:.4f}\n")

            f.write(f"\n\nBEST MODEL: {self.best_model_name}\n")
            f.write(f"  Primary metric (Recall): {self.results[self.best_model_name]['Recall']:.4f}\n")
            f.write(f"  ROC-AUC: {self.results[self.best_model_name]['ROC-AUC']:.4f}\n")
            f.write(f"  Fraud catch rate: {self.results[self.best_model_name]['True Positives']} / "
                    f"{self.results[self.best_model_name]['True Positives'] + self.results[self.best_model_name]['False Negatives']} frauds caught\n")

        print(f"  ✓ Report: {report_path}")

        # Save metadata
        metadata_path = output_dir / "metadata.txt"
        with open(metadata_path, 'w') as f:
            f.write(f"best_model: {self.best_model_name}\n")
            f.write(f"total_models_trained: {len(self.models)}\n")
            f.write(f"test_set_size: {len(self.X_test)}\n")
            f.write(f"fraud_rate_test: {100*self.y_test.mean():.2f}%\n")
            f.write(f"best_recall: {self.results[self.best_model_name]['Recall']:.4f}\n")
            f.write(f"best_roc_auc: {self.results[self.best_model_name]['ROC-AUC']:.4f}\n")
        print(f"  ✓ Metadata: {metadata_path}")

    def run(self, input_dir: Path = None, output_dir: Path = None):
        """Execute complete model training pipeline."""
        print("\n" + "=" * 80)
        print("PHASE 5: MODEL TRAINING & EVALUATION")
        print("=" * 80)

        X, y = self.load_data(input_dir)
        self.split_data(X, y)

        self.train_logistic_regression()
        self.train_random_forest()
        self.train_xgboost()

        self.select_best_model()
        self.save_outputs(output_dir)

        print("\n" + "=" * 80)
        print("✓ MODEL TRAINING COMPLETE")
        print("=" * 80)
        print(f"\n{len(self.models)} models trained and evaluated")
        print(f"Best model selected: {self.best_model_name}")
        print(f"\nNext phase: MLflow Integration for experiment tracking")
        print()


if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.run()
