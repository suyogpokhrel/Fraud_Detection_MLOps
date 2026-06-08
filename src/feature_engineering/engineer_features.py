"""
Phase 4: Feature Engineering Pipeline
Loads preprocessed data from Redis (fast) or CSV (fallback),
creates domain-specific fraud detection features,
and caches results in Redis for the model training step.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

from src.cache.redis_client import load_dataframe, save_dataframe


class FeatureEngineer:

    def __init__(self):
        self.X = None
        self.y = None
        self.X_engineered = None
        self.feature_descriptions = {}

    def load_data(self, input_dir: Path = None):
        """Load preprocessed data from Redis if available, otherwise from CSV."""
        if input_dir is None:
            input_dir = repo_root / "data" / "preprocessed"

        print("\n[1/5] Loading preprocessed data...")

        # Try Redis first
        df = load_dataframe("fraud:preprocessed")
        if df is not None:
            print(f"✓ Loaded {len(df)} rows from Redis (fast cache)")
            self.y = df["isFraud"].copy()
            self.X = df.drop(columns=["isFraud"]).copy()
            return

        # Fall back to CSV
        print("  Redis miss — loading from CSV...")
        X_path = input_dir / "X_preprocessed.csv"
        y_path = input_dir / "y_preprocessed.csv"
        if not X_path.exists() or not y_path.exists():
            raise FileNotFoundError(f"Preprocessed data not found in {input_dir}")
        self.X = pd.read_csv(X_path)
        self.y = pd.read_csv(y_path).squeeze()
        print(f"✓ Loaded {len(self.X)} rows from CSV")

    def create_balance_features(self):
        print("\n[2/5] Creating balance-based features...")

        self.X_engineered["sender_balance_change"] = (
            self.X["newbalanceOrig"] - self.X["oldbalanceOrg"]
        )
        self.X_engineered["receiver_balance_change"] = (
            self.X["newbalanceDest"] - self.X["oldbalanceDest"]
        )
        self.X_engineered["sender_balance_change_abs"] = np.abs(
            self.X_engineered["sender_balance_change"]
        )
        self.X_engineered["receiver_balance_change_abs"] = np.abs(
            self.X_engineered["receiver_balance_change"]
        )
        self.X_engineered["sender_balance_change_ratio"] = np.where(
            self.X["oldbalanceOrg"] != 0,
            self.X_engineered["sender_balance_change"] / np.abs(self.X["oldbalanceOrg"] + 1e-10),
            0
        )
        self.X_engineered["receiver_balance_change_ratio"] = np.where(
            self.X["oldbalanceDest"] != 0,
            self.X_engineered["receiver_balance_change"] / np.abs(self.X["oldbalanceDest"] + 1e-10),
            0
        )
        print("✓ Created 6 balance-based features")

    def create_transaction_amount_features(self):
        print("\n[3/5] Creating transaction amount features...")

        self.X_engineered["amount_ratio_sender_balance"] = np.where(
            self.X["oldbalanceOrg"] > 0,
            self.X["amount"] / (self.X["oldbalanceOrg"] + 1e-10),
            0
        )
        self.X_engineered["amount_ratio_receiver_balance"] = np.where(
            self.X["oldbalanceDest"] > 0,
            self.X["amount"] / (self.X["oldbalanceDest"] + 1e-10),
            0
        )
        self.X_engineered["sender_depletes_account"] = (
            np.abs(self.X_engineered["sender_balance_change"]) >=
            np.abs(self.X["oldbalanceOrg"]) * 0.99
        ).astype(int)
        self.X_engineered["receiver_had_zero_balance"] = (
            self.X["oldbalanceDest"] == 0
        ).astype(int)
        amount_q90 = self.X["amount"].quantile(0.90)
        self.X_engineered["is_high_value_transaction"] = (
            self.X["amount"] >= amount_q90
        ).astype(int)
        print("✓ Created 5 transaction amount features")

    def create_account_type_features(self):
        print("\n[4/5] Creating account type interaction features...")

        self.X_engineered["current_to_savings_transfer"] = (
            (self.X["acct_type_Current"] == 1) &
            (self.X["type_TRANSFER"] == True)
        ).astype(int)
        self.X_engineered["cashout_with_unusual_login"] = (
            (self.X["type_CASH_OUT"] == True) &
            (self.X["unusuallogin"] > 5)
        ).astype(int)
        self.X_engineered["high_fraud_branch_high_amount"] = (
            (self.X["branch_fraud_rate"] > 0.01) &
            (self.X["amount"] > self.X["amount"].quantile(0.75))
        ).astype(int)
        self.X_engineered["flagged_and_unusual"] = (
            (self.X["isFlaggedFraud"] == 1) &
            (self.X["unusuallogin"] > 3)
        ).astype(int)
        print("✓ Created 4 account type interaction features")

    def create_derived_features(self):
        print("\n[5/5] Creating derived signal features...")

        self.X_engineered["suspicion_score"] = (
            self.X_engineered["sender_depletes_account"] +
            self.X_engineered["receiver_had_zero_balance"] +
            self.X_engineered["cashout_with_unusual_login"] +
            self.X_engineered["high_fraud_branch_high_amount"] +
            self.X_engineered["flagged_and_unusual"]
        )
        self.X_engineered["risk_exposure_sender"] = (
            np.abs(self.X_engineered["sender_balance_change_ratio"]) *
            self.X_engineered["amount_ratio_sender_balance"]
        )
        self.X_engineered["total_money_flow"] = (
            np.abs(self.X_engineered["sender_balance_change"]) +
            np.abs(self.X_engineered["receiver_balance_change"])
        )
        print("✓ Created 3 derived signal features")

    def validate_engineered_features(self):
        nan_counts = self.X_engineered.isna().sum()
        if nan_counts.sum() > 0:
            print(f"  ⚠ WARNING: {nan_counts.sum()} NaN values found")
        else:
            print(f"\n✓ No NaN values in {len(self.X_engineered.columns)} engineered features")
        inf_count = np.isinf(self.X_engineered.select_dtypes(include=[np.number])).sum().sum()
        if inf_count > 0:
            print(f"  ⚠ WARNING: {inf_count} infinite values found")
        else:
            print("✓ No infinite values")

    def save_outputs(self, output_dir: Path = None):
        if output_dir is None:
            output_dir = repo_root / "data" / "engineered_features"

        print(f"\n✓ Saving outputs to {output_dir}...")
        output_dir.mkdir(parents=True, exist_ok=True)

        X_combined = pd.concat([self.X, self.X_engineered], axis=1)
        X_combined.to_csv(output_dir / "X_engineered.csv", index=False)
        self.X_engineered.to_csv(output_dir / "X_engineered_only.csv", index=False)
        self.y.to_csv(output_dir / "y_engineered.csv", index=False, header=True)

        print(f"  ✓ CSV files saved — shape: {X_combined.shape}")

        # Cache engineered data for model training step
        combined = X_combined.copy()
        combined["isFraud"] = self.y.values
        saved = save_dataframe("fraud:engineered", combined)
        if saved:
            print(f"  ✓ Cached engineered data to Redis key 'fraud:engineered'")
        else:
            print("  Redis unavailable — model training will use CSV")

    def run(self, input_dir: Path = None, output_dir: Path = None):
        print("\n" + "="*70)
        print("PHASE 4: FEATURE ENGINEERING PIPELINE")
        print("="*70)

        self.load_data(input_dir)
        self.X_engineered = pd.DataFrame(index=self.X.index)

        self.create_balance_features()
        self.create_transaction_amount_features()
        self.create_account_type_features()
        self.create_derived_features()
        self.validate_engineered_features()
        self.save_outputs(output_dir)

        print("\n✓ FEATURE ENGINEERING COMPLETE")
        print(f"Original features: {len(self.X.columns)}")
        print(f"Engineered features: {len(self.X_engineered.columns)}")
        print(f"Total features: {len(self.X.columns) + len(self.X_engineered.columns)}")


if __name__ == "__main__":
    engineer = FeatureEngineer()
    engineer.run()
