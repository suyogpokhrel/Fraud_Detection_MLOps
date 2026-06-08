"""
Phase 3: Data Preprocessing Pipeline
Loads raw transactions from Redis (fast) or MariaDB (fallback),
applies preprocessing transformations, saves ML-ready features,
and caches results in Redis for the next pipeline step.
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import warnings

warnings.filterwarnings('ignore')

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

from database.db_connection import get_connection
from src.cache.redis_client import load_dataframe, save_dataframe


class PreprocessingPipeline:

    def __init__(self):
        self.df_original = None
        self.df_processed = None
        self.X_train = None
        self.y_train = None
        self.scaler = None
        self.feature_names = None
        self.numeric_columns = [
            "step", "amount", "oldbalanceOrg", "newbalanceOrig",
            "oldbalanceDest", "newbalanceDest", "unusuallogin"
        ]
        self.categorical_columns_low = ["type", "acct_type"]
        self.categorical_columns_high = ["branch"]
        self.target_column = "isFraud"

    def load_data(self):
        """Load data from Redis if available, otherwise fall back to MariaDB."""
        print("\n[1/7] Loading data...")

        # Try Redis first
        df = load_dataframe("fraud:raw_data")
        if df is not None:
            print(f"✓ Loaded {len(df)} rows from Redis (fast cache)")
            self.df_original = df
            return

        # Fall back to MariaDB
        print("  Redis miss — loading from MariaDB...")
        conn = get_connection()
        try:
            self.df_original = pd.read_sql("SELECT * FROM processed_transactions", conn)
            print(f"✓ Loaded {len(self.df_original)} rows from MariaDB")
        finally:
            conn.close()

    def detect_outliers_iqr(self, series, column_name):
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outlier_count = ((series < lower_bound) | (series > upper_bound)).sum()
        if outlier_count > 0:
            print(f"  └─ {column_name}: {outlier_count} outliers (bounds: {lower_bound:.2f}, {upper_bound:.2f})")
        return lower_bound, upper_bound

    def handle_outliers(self):
        print("\n[2/7] Handling outliers (IQR method)...")
        self.df_processed = self.df_original.copy()
        for col in self.numeric_columns:
            if col in self.df_processed.columns:
                lower, upper = self.detect_outliers_iqr(self.df_processed[col], col)
                self.df_processed[col] = self.df_processed[col].clip(lower=lower, upper=upper)
        print("✓ Outliers capped (rows preserved)")

    def scale_numeric_features(self):
        print("\n[3/7] Scaling numeric features (StandardScaler)...")
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        X_numeric = self.df_processed[self.numeric_columns].copy()
        self.df_processed[self.numeric_columns] = self.scaler.fit_transform(X_numeric)
        print(f"✓ Scaled {len(self.numeric_columns)} numeric columns")

    def encode_categorical_features(self):
        print("\n[4/7] Encoding categorical features...")
        X_onehot_list = []

        for col in self.categorical_columns_low:
            unique_vals = self.df_processed[col].unique()
            print(f"  └─ One-Hot encoding {col} ({len(unique_vals)} unique values)")
            one_hot = pd.get_dummies(self.df_processed[col], prefix=col, drop_first=False)
            X_onehot_list.append(one_hot)

        branch_fraud_rate = self.df_processed.groupby("branch")[self.target_column].mean()
        print(f"  └─ Target encoding branch ({len(branch_fraud_rate)} unique values)")
        branch_encoded = self.df_processed["branch"].map(branch_fraud_rate)
        X_onehot_list.append(pd.DataFrame({"branch_fraud_rate": branch_encoded}))

        X_encoded_categorical = pd.concat(X_onehot_list, axis=1)
        self.df_processed = self.df_processed.drop(
            columns=self.categorical_columns_low + self.categorical_columns_high
        )
        self.df_processed = pd.concat([self.df_processed, X_encoded_categorical], axis=1)
        print(f"✓ Categorical encoding complete")

    def prepare_final_features(self):
        print("\n[5/7] Preparing final features...")
        cols_to_drop = [
            "transaction_id", "nameOrig", "nameDest",
            "date_of_transaction", "time_of_day", "datetime"
        ]
        all_cols = self.df_processed.columns.tolist()
        feature_cols = [
            col for col in all_cols
            if col != self.target_column and col not in cols_to_drop
        ]
        self.X_train = self.df_processed[feature_cols].copy()
        self.y_train = self.df_processed[self.target_column].copy()
        self.feature_names = feature_cols
        print(f"✓ Features: {len(self.X_train.columns)} columns, {len(self.X_train)} rows")

    def validate_output(self):
        print("\n[6/7] Validating preprocessed data...")
        nan_counts = self.X_train.isna().sum()
        if (nan_counts > 0).any():
            print(f"  ⚠ WARNING: {nan_counts.sum()} NaN values found")
        else:
            print("  ✓ No NaN values")
        fraud_dist = self.y_train.value_counts().sort_index()
        for label, count in fraud_dist.items():
            pct = round(count / len(self.y_train) * 100, 2)
            print(f"  ✓ isFraud={label}: {count} rows ({pct}%)")
        print("✓ Validation passed")

    def save_outputs(self, output_dir: Path):
        print(f"\n[7/7] Saving outputs to {output_dir}...")
        output_dir.mkdir(parents=True, exist_ok=True)

        self.X_train.to_csv(output_dir / "X_preprocessed.csv", index=False)
        self.y_train.to_csv(output_dir / "y_preprocessed.csv", index=False)
        joblib.dump(self.scaler, output_dir / "scaler.joblib")

        with open(output_dir / "feature_names.txt", 'w') as f:
            for fname in self.feature_names:
                f.write(fname + "\n")

        print("  ✓ CSV files saved")

        # Cache preprocessed data for feature engineering step
        combined = self.X_train.copy()
        combined[self.target_column] = self.y_train.values
        saved = save_dataframe("fraud:preprocessed", combined)
        if saved:
            print(f"  ✓ Cached preprocessed data to Redis key 'fraud:preprocessed'")
        else:
            print("  Redis unavailable — feature engineering will use CSV")

    def run(self, output_dir: Path = None):
        if output_dir is None:
            output_dir = repo_root / "data" / "preprocessed"

        print("\n" + "="*70)
        print("PHASE 3: DATA PREPROCESSING PIPELINE")
        print("="*70)

        self.load_data()
        self.handle_outliers()
        self.scale_numeric_features()
        self.encode_categorical_features()
        self.prepare_final_features()
        self.validate_output()
        self.save_outputs(output_dir)

        print("\n✓ PREPROCESSING COMPLETE")
        print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    pipeline = PreprocessingPipeline()
    pipeline.run()
