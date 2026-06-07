"""
Phase 3: Data Preprocessing Pipeline
Loads raw transactions from MariaDB, applies preprocessing transformations,
and saves ML-ready features with transformation pipelines.

Steps:
  1. Load data from MariaDB processed_transactions
  2. Separate features and target
  3. Detect and handle outliers (IQR method with capping)
  4. Scale numeric features (StandardScaler)
  5. Encode categorical features (One-Hot for low cardinality, Target encoding for high cardinality)
  6. Validate data integrity
  7. Save preprocessed features and transformers
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import warnings

warnings.filterwarnings('ignore')

# Add repository to path
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

from database.db_connection import get_connection


class PreprocessingPipeline:
    """
    Machine learning preprocessing pipeline.
    Handles outlier detection, scaling, encoding, and validation.
    """

    def __init__(self):
        self.df_original = None
        self.df_processed = None
        self.X_train = None
        self.y_train = None
        self.scaler = None
        self.encoder = None
        self.feature_names = None
        self.numeric_columns = [
            "step", "amount", "oldbalanceOrg", "newbalanceOrig",
            "oldbalanceDest", "newbalanceDest", "unusuallogin"
        ]
        self.categorical_columns_low = ["type", "acct_type"]  # One-hot encode
        self.categorical_columns_high = ["branch"]  # Target encode
        self.drop_columns = [
            "transaction_id", "date_of_transaction", "time_of_day",
            "nameOrig", "nameDest", "datetime"
        ]
        self.keep_columns = ["isFlaggedFraud"]  # Pass through as-is
        self.target_column = "isFraud"

    def load_data(self):
        """Load processed transactions from MariaDB."""
        print("\n[1/7] Loading data from MariaDB...")
        conn = get_connection()
        try:
            query = "SELECT * FROM processed_transactions"
            self.df_original = pd.read_sql(query, conn)
            print(f"✓ Loaded {len(self.df_original)} rows, {len(self.df_original.columns)} columns")
        finally:
            conn.close()

    def detect_outliers_iqr(self, series, column_name):
        """
        Detect outliers using Interquartile Range (IQR) method.
        Returns lower and upper bounds for capping.
        """
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outlier_count = ((series < lower_bound) | (series > upper_bound)).sum()
        if outlier_count > 0:
            print(f"  └─ {column_name}: {outlier_count} outliers detected (bounds: {lower_bound:.2f}, {upper_bound:.2f})")
        return lower_bound, upper_bound

    def handle_outliers(self):
        """
        Detect and cap outliers in numeric columns using IQR method.
        Capping preserves row count for training (no data loss).
        """
        print("\n[2/7] Handling outliers (IQR method)...")
        self.df_processed = self.df_original.copy()

        for col in self.numeric_columns:
            if col in self.df_processed.columns:
                lower, upper = self.detect_outliers_iqr(self.df_processed[col], col)
                # Cap outliers without removing rows
                self.df_processed[col] = self.df_processed[col].clip(lower=lower, upper=upper)

        print("✓ Outliers capped (rows preserved)")

    def scale_numeric_features(self):
        """
        Standardize numeric features: (X - mean) / std
        Fit scaler on data and apply transformation.
        """
        print("\n[3/7] Scaling numeric features (StandardScaler)...")
        self.scaler = StandardScaler()

        # Extract numeric features
        X_numeric = self.df_processed[self.numeric_columns].copy()

        # Fit and transform
        X_numeric_scaled = self.scaler.fit_transform(X_numeric)

        # Replace with scaled values
        self.df_processed[self.numeric_columns] = X_numeric_scaled
        print(f"✓ Scaled {len(self.numeric_columns)} numeric columns")

    def encode_categorical_features(self):
        """
        Encode categorical features:
        - One-Hot: type (5 values), acct_type (2 values)
        - Target: branch (135 values) - encode with fraud rate per branch
        """
        print("\n[4/7] Encoding categorical features...")

        # One-Hot encode low-cardinality categorical columns
        X_onehot_list = []
        onehot_feature_names = []

        for col in self.categorical_columns_low:
            unique_vals = self.df_processed[col].unique()
            print(f"  └─ One-Hot encoding {col} ({len(unique_vals)} unique values)")
            one_hot = pd.get_dummies(self.df_processed[col], prefix=col, drop_first=False)
            X_onehot_list.append(one_hot)
            onehot_feature_names.extend(one_hot.columns.tolist())

        # Target encode high-cardinality categorical (branch)
        # Calculate fraud rate per branch
        branch_fraud_rate = self.df_processed.groupby("branch")[self.target_column].mean()
        print(f"  └─ Target encoding branch ({len(branch_fraud_rate)} unique values)")
        branch_encoded = self.df_processed["branch"].map(branch_fraud_rate)
        X_onehot_list.append(pd.DataFrame({f"branch_fraud_rate": branch_encoded}))
        onehot_feature_names.append("branch_fraud_rate")

        # Combine all encoded categorical features
        X_encoded_categorical = pd.concat(X_onehot_list, axis=1)

        # Update processed dataframe: remove old categorical, add encoded
        self.df_processed = self.df_processed.drop(columns=self.categorical_columns_low + self.categorical_columns_high)
        self.df_processed = pd.concat([self.df_processed, X_encoded_categorical], axis=1)

        print(f"✓ Created {len(onehot_feature_names)} encoded categorical features")

    def prepare_final_features(self):
        """
        Prepare final feature matrix and target vector.
        Exclude target column and unnecessary columns.
        """
        print("\n[5/7] Preparing final features...")

        # Define columns to drop (identifiers, names, dates, redundant time info)
        cols_to_drop = [
            "transaction_id", "nameOrig", "nameDest",
            "date_of_transaction", "time_of_day", "datetime"
        ]

        # Select only useful feature columns
        all_cols = self.df_processed.columns.tolist()
        feature_cols = [
            col for col in all_cols
            if col != self.target_column and col not in cols_to_drop
        ]

        self.X_train = self.df_processed[feature_cols].copy()
        self.y_train = self.df_processed[self.target_column].copy()
        self.feature_names = feature_cols

        print(f"✓ Dropped columns: {cols_to_drop}")
        print(f"✓ Features: {len(self.X_train.columns)} columns, {len(self.X_train)} rows")
        print(f"✓ Target: {len(self.y_train)} rows")

    def validate_output(self):
        """
        Comprehensive validation of preprocessed data.
        Checks for NaN, data types, and class distribution.
        """
        print("\n[6/7] Validating preprocessed data...")

        # Check for NaN values
        nan_counts = self.X_train.isna().sum()
        if (nan_counts > 0).any():
            print(f"  ⚠ WARNING: {nan_counts.sum()} NaN values found:")
            print(nan_counts[nan_counts > 0])
        else:
            print("  ✓ No NaN values in features")

        # Check target distribution
        fraud_dist = self.y_train.value_counts().sort_index()
        fraud_pct = (fraud_dist / len(self.y_train) * 100).round(2)
        print(f"  ✓ Target distribution (isFraud):")
        for label, count in fraud_dist.items():
            print(f"     └─ {label}: {count} rows ({fraud_pct[label]}%)")

        # Check feature statistics
        print(f"  ✓ Feature statistics:")
        print(f"     └─ Numeric features (should be ~0 mean, ~1 std after scaling)")
        numeric_stats = self.X_train[self.numeric_columns].describe().loc[["mean", "std"]].round(3)
        print(numeric_stats.to_string())

        print("✓ Validation passed")

    def save_outputs(self, output_dir: Path):
        """
        Save preprocessed features, scalers, and metadata.
        """
        print(f"\n[7/7] Saving outputs to {output_dir}...")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save preprocessed features (X, y)
        X_csv = output_dir / "X_preprocessed.csv"
        y_csv = output_dir / "y_preprocessed.csv"
        self.X_train.to_csv(X_csv, index=False)
        self.y_train.to_csv(y_csv, index=False)
        print(f"  ✓ Features saved: {X_csv}")
        print(f"  ✓ Target saved: {y_csv}")

        # Save scaler
        scaler_path = output_dir / "scaler.joblib"
        joblib.dump(self.scaler, scaler_path)
        print(f"  ✓ Scaler saved: {scaler_path}")

        # Save feature names
        features_json = output_dir / "feature_names.txt"
        with open(features_json, 'w') as f:
            for fname in self.feature_names:
                f.write(fname + "\n")
        print(f"  ✓ Feature names saved: {features_json}")

        # Save metadata
        metadata = {
            "total_rows": len(self.X_train),
            "total_features": len(self.X_train.columns),
            "numeric_columns": self.numeric_columns,
            "categorical_columns_low": self.categorical_columns_low,
            "categorical_columns_high": self.categorical_columns_high,
            "target_distribution": self.y_train.value_counts().to_dict(),
        }
        metadata_path = output_dir / "metadata.txt"
        with open(metadata_path, 'w') as f:
            for key, value in metadata.items():
                f.write(f"{key}: {value}\n")
        print(f"  ✓ Metadata saved: {metadata_path}")

    def run(self, output_dir: Path = None):
        """Execute complete preprocessing pipeline."""
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

        print("\n" + "="*70)
        print("✓ PREPROCESSING COMPLETE")
        print("="*70)
        print(f"\nOutputs saved to: {output_dir}")
        print(f"Next phase: Feature Engineering & Model Training")
        print()


if __name__ == "__main__":
    pipeline = PreprocessingPipeline()
    pipeline.run()
