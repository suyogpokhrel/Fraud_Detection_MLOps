
from pathlib import Path
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib
import warnings
warnings.filterwarnings('ignore')

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

from src.cache.redis_client import load_dataframe, save_dataframe, save_json, load_json


class PreprocessingPipeline:

    # Columns we actually use — nameOrig/nameDest dropped (too many unique values)
    NUMERIC_COLS  = ["step", "amount", "oldbalanceOrg", "newbalanceOrig",
                     "oldbalanceDest", "newbalanceDest"]
    CATEGORIC_COLS = ["type"]
    TARGET_COL     = "isFraud"

    def __init__(self):
        self.scaler      = StandardScaler()
        self.encoder     = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.feature_names = []

    # ── helpers ───────────────────────────────────────────────────────────────

    def _load_data(self, repo_root: Path) -> pd.DataFrame:
        """Load from Redis, then MariaDB ColumnStore, then CSV."""
        # 1. Try Redis
        df = load_dataframe("pipeline:processed")
        if df is not None:
            print(f"  Loaded from Redis: {df.shape}")
            return df

        # 2. Try MariaDB ColumnStore
        try:
            import subprocess
            result = subprocess.run(
                ["mariadb", "-u", "fraud_user", "-pfraud_password",
                 "fraud_detection", "-e",
                 "SELECT step,type,amount,nameOrig,oldbalanceOrg,newbalanceOrig,nameDest,oldbalanceDest,newbalanceDest,isFraud FROM processed_transactions LIMIT 1;"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0 and "step" in result.stdout:
                import io
                full = subprocess.run(
                    ["mariadb", "-u", "fraud_user", "-pfraud_password",
                     "--batch", "--silent", "fraud_detection", "-e",
                     "SELECT step,type,amount,nameOrig,oldbalanceOrg,newbalanceOrig,nameDest,oldbalanceDest,newbalanceDest,isFraud FROM processed_transactions;"],
                    capture_output=True, text=True, timeout=60
                )
                cols = ["step","type","amount","nameOrig","oldbalanceOrg",
                        "newbalanceOrig","nameDest","oldbalanceDest","newbalanceDest","isFraud"]
                df = pd.read_csv(io.StringIO(full.stdout), sep="	", names=cols)
                print(f"  Loaded from MariaDB ColumnStore: {df.shape}")
                save_dataframe("pipeline:processed", df)
                return df
        except Exception as e:
            print(f"  MariaDB ColumnStore unavailable: {e}")

        # 3. Fall back to CSV
        csv_path = repo_root / "data" / "processed" / "processed_data.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Processed CSV not found: {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"  Loaded from CSV: {df.shape}")
        save_dataframe("pipeline:processed", df)
        return df

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop unused columns, remove nulls."""
        keep = self.NUMERIC_COLS + self.CATEGORIC_COLS + [self.TARGET_COL]
        df = df[[c for c in keep if c in df.columns]].copy()
        before = len(df)
        df = df.dropna()
        df[self.TARGET_COL] = df[self.TARGET_COL].astype(int)
        print(f"  Rows after cleaning: {len(df):,}  (dropped {before - len(df)})")
        return df

    # ── main run ──────────────────────────────────────────────────────────────

    def run(self, output_dir: Path = None):
        if output_dir is None:
            output_dir = repo_root / "data" / "preprocessed"
        output_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 60)
        print("PHASE 3: PREPROCESSING")
        print("=" * 60)

        # Load
        print("\n[1/4] Loading data...")
        df = self._load_data(repo_root)

        # Clean
        print("\n[2/4] Cleaning...")
        df = self._clean(df)
        print(f"  Fraud rate: {df[self.TARGET_COL].mean()*100:.2f}%")

        # Fit & transform
        print("\n[3/4] Scaling and encoding...")
        X = df[self.NUMERIC_COLS + self.CATEGORIC_COLS]
        y = df[self.TARGET_COL]

        # Fit scaler on numeric
        X_numeric = self.scaler.fit_transform(df[self.NUMERIC_COLS])

        # Fit encoder on categorical
        X_cat = self.encoder.fit_transform(df[self.CATEGORIC_COLS])
        cat_names = list(self.encoder.get_feature_names_out(self.CATEGORIC_COLS))

        # Combine
        X_preprocessed = np.hstack([X_numeric, X_cat])
        self.feature_names = self.NUMERIC_COLS + cat_names

        X_df = pd.DataFrame(X_preprocessed, columns=self.feature_names)
        y_df = pd.Series(y.values, name=self.TARGET_COL)

        print(f"  Features: {len(self.feature_names)}")
        print(f"  Feature names: {self.feature_names}")

        # Save to disk
        print("\n[4/4] Saving outputs...")
        X_df.to_csv(output_dir / "X_preprocessed.csv", index=False)
        y_df.to_csv(output_dir / "y_preprocessed.csv", index=False)
        joblib.dump(self.scaler,  output_dir / "scaler.joblib")
        joblib.dump(self.encoder, output_dir / "encoder.joblib")

        # Save feature names
        with open(output_dir / "feature_names.txt", "w") as f:
            for name in self.feature_names:
                f.write(name + "\n")

        # Save metadata
        metadata = {
            "n_samples":      int(len(X_df)),
            "n_features":     int(len(self.feature_names)),
            "fraud_rate_pct": float(round(y_df.mean() * 100, 4)),
            "numeric_cols":   self.NUMERIC_COLS,
            "categoric_cols": self.CATEGORIC_COLS,
            "feature_names":  self.feature_names,
        }
        save_json("pipeline:preprocessed_meta", metadata)

        # Cache preprocessed data in Redis
        save_dataframe("pipeline:X_preprocessed", X_df)
        save_dataframe("pipeline:y_preprocessed", y_df.to_frame())

        print(f"  X_preprocessed.csv : {X_df.shape}")
        print(f"  y_preprocessed.csv : {y_df.shape}")
        print(f"  scaler.joblib saved")
        print(f"  encoder.joblib saved")
        print(f"\n  Null check: {X_df.isnull().sum().sum()} nulls")
        print("\n✓ PREPROCESSING COMPLETE")

        return X_df, y_df


if __name__ == "__main__":
    pipeline = PreprocessingPipeline()
    pipeline.run()
