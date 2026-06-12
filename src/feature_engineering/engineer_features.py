
from pathlib import Path
import sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

from src.cache.redis_client import load_dataframe, save_dataframe, load_json


class FeatureEngineer:

    def __init__(self):
        self.X_engineered = None
        self.y = None

    # ── helpers ───────────────────────────────────────────────────────────────

    def _load_data(self, input_dir: Path):
        """Load preprocessed data from Redis or CSV."""
        X = load_dataframe("pipeline:X_preprocessed")
        y = load_dataframe("pipeline:y_preprocessed")

        if X is not None and y is not None:
            print(f"  Loaded X from Redis: {X.shape}")
            print(f"  Loaded y from Redis: {y.shape}")
            return X, y.iloc[:, 0]

        print("  Redis unavailable, loading from CSV...")
        X_path = input_dir / "X_preprocessed.csv"
        y_path = input_dir / "y_preprocessed.csv"

        if not X_path.exists():
            raise FileNotFoundError(f"Preprocessed data not found: {X_path}")

        X = pd.read_csv(X_path)
        y = pd.read_csv(y_path).iloc[:, 0]
        print(f"  Loaded X from CSV: {X.shape}")
        print(f"  Loaded y from CSV: {y.shape}")
        return X, y

    def _load_raw_for_engineering(self, repo_root: Path) -> pd.DataFrame:
        """Load raw numeric columns needed for engineering."""
        raw = load_dataframe("pipeline:processed")
        if raw is None:
            csv_path = repo_root / "data" / "processed" / "processed_data.csv"
            raw = pd.read_csv(csv_path)
            print(f"  Loaded raw from CSV: {raw.shape}")
        else:
            print(f"  Loaded raw from Redis: {raw.shape}")
        return raw

    # ── feature creation ──────────────────────────────────────────────────────

    def _create_features(self, X: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
        """
        Create engineered features from raw balance/amount columns.
        X already contains scaled numeric + one-hot type features.
        We add engineered features using the raw (unscaled) values.
        """
        raw = raw.reset_index(drop=True)
        X   = X.reset_index(drop=True)

        eng = pd.DataFrame(index=X.index)

        # Balance change features
        eng["sender_balance_change"]    = raw["newbalanceOrig"]  - raw["oldbalanceOrg"]
        eng["receiver_balance_change"]  = raw["newbalanceDest"]  - raw["oldbalanceDest"]

        # Amount ratio features
        eng["amount_ratio_sender"]   = raw["amount"] / (raw["oldbalanceOrg"]  + 1e-9)
        eng["amount_ratio_receiver"] = raw["amount"] / (raw["oldbalanceDest"] + 1e-9)

        # Binary fraud signal features
        eng["sender_depletes_account"]   = (
            (raw["newbalanceOrig"] == 0) & (raw["oldbalanceOrg"] > 0)
        ).astype(int)

        eng["receiver_had_zero_balance"] = (raw["oldbalanceDest"] == 0).astype(int)

        # Transaction type signal
        eng["is_transfer_or_cashout"] = raw["type"].isin(["TRANSFER", "CASH_OUT"]).astype(int)

        # Balance mismatch — in real fraud the expected change doesn't match actual
        # Expected: oldbalance - amount = newbalance (for sender)
        eng["balance_mismatch_sender"]   = abs(
            (raw["oldbalanceOrg"] - raw["amount"]) - raw["newbalanceOrig"]
        )
        eng["balance_mismatch_receiver"] = abs(
            (raw["oldbalanceDest"] + raw["amount"]) - raw["newbalanceDest"]
        )

        # Clip extreme ratios to avoid inf
        eng["amount_ratio_sender"]   = eng["amount_ratio_sender"].clip(0, 1000)
        eng["amount_ratio_receiver"] = eng["amount_ratio_receiver"].clip(0, 1000)

        print(f"  Engineered features created: {list(eng.columns)}")
        return eng

    # ── main run ──────────────────────────────────────────────────────────────

    def run(self, input_dir: Path = None, output_dir: Path = None):
        if input_dir is None:
            input_dir  = repo_root / "data" / "preprocessed"
        if output_dir is None:
            output_dir = repo_root / "data" / "engineered_features"
        output_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 60)
        print("PHASE 4: FEATURE ENGINEERING")
        print("=" * 60)

        # Load preprocessed
        print("\n[1/4] Loading preprocessed data...")
        X, y = self._load_data(input_dir)

        # Load raw for engineering
        print("\n[2/4] Loading raw data for feature engineering...")
        raw = self._load_raw_for_engineering(repo_root)

        # Align sizes
        min_len = min(len(X), len(raw))
        X   = X.iloc[:min_len].reset_index(drop=True)
        y   = y.iloc[:min_len].reset_index(drop=True)
        raw = raw.iloc[:min_len].reset_index(drop=True)

        # Create engineered features
        print("\n[3/4] Creating engineered features...")
        eng = self._create_features(X, raw)

        # Combine preprocessed + engineered
        X_final = pd.concat([X, eng], axis=1)
        self.X_engineered = X_final
        self.y = y

        print(f"\n  Preprocessed features : {X.shape[1]}")
        print(f"  Engineered features   : {eng.shape[1]}")
        print(f"  Total features        : {X_final.shape[1]}")
        print(f"  Total samples         : {X_final.shape[0]:,}")
        print(f"  Null values           : {X_final.isnull().sum().sum()}")

        # Save
        print("\n[4/4] Saving outputs...")
        X_final.to_csv(output_dir / "X_engineered.csv", index=False)
        y.to_csv(output_dir / "y_engineered.csv", index=False)

        # Save feature list
        with open(output_dir / "all_features.txt", "w") as f:
            for i, name in enumerate(X_final.columns, 1):
                f.write(f"{i:3d}. {name}\n")

        # Cache in Redis
        save_dataframe("pipeline:X_engineered", X_final)
        save_dataframe("pipeline:y_engineered", y.to_frame())

        print(f"  X_engineered.csv saved: {X_final.shape}")
        print(f"  y_engineered.csv saved: {y.shape}")
        print(f"  all_features.txt saved")
        print("\n✓ FEATURE ENGINEERING COMPLETE")

        return X_final, y


if __name__ == "__main__":
    engineer = FeatureEngineer()
    engineer.run()
