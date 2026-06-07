"""
Phase 4: Feature Engineering Pipeline
Creates domain-specific fraud detection features from preprocessed transactions.

Feature Categories:
  1. Balance-based features: Account balance changes, ratios, deltas
  2. Transaction-based features: Amount ratios, anomaly flags
  3. Account behavior features: Risk indicators, account type interactions
  4. Derived interaction features: Complex fraud signals
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

# Add repository to path
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))


class FeatureEngineer:
    """
    Advanced feature engineering for fraud detection.
    Creates domain-specific features that capture fraud patterns.
    """

    def __init__(self):
        self.X = None
        self.y = None
        self.X_engineered = None
        self.feature_descriptions = {}
        self.input_features = [
            "step", "amount", "oldbalanceOrg", "newbalanceOrig",
            "oldbalanceDest", "newbalanceDest", "unusuallogin", "isFlaggedFraud",
            "type_CASH_IN", "type_CASH_OUT", "type_DEBIT", "type_PAYMENT", "type_TRANSFER",
            "acct_type_Current", "acct_type_Savings", "branch_fraud_rate"
        ]

    def load_data(self, input_dir: Path = None):
        """Load preprocessed features from Phase 3."""
        if input_dir is None:
            input_dir = repo_root / "data" / "preprocessed"

        print("\n[1/5] Loading preprocessed data...")
        X_path = input_dir / "X_preprocessed.csv"
        y_path = input_dir / "y_preprocessed.csv"

        if not X_path.exists() or not y_path.exists():
            raise FileNotFoundError(f"Preprocessed data not found in {input_dir}")

        self.X = pd.read_csv(X_path)
        self.y = pd.read_csv(y_path).squeeze()

        print(f"✓ Loaded features: {self.X.shape[0]} rows × {self.X.shape[1]} columns")
        print(f"✓ Loaded target: {len(self.y)} rows")

    def create_balance_features(self):
        """
        Create features based on account balance changes.
        
        Rationale:
        - Fraudsters often drain accounts (large negative balance change)
        - Unusual balance modifications indicate suspicious activity
        """
        print("\n[2/5] Creating balance-based features...")

        # Sender balance change
        self.X_engineered["sender_balance_change"] = (
            self.X["newbalanceOrig"] - self.X["oldbalanceOrg"]
        )
        self.feature_descriptions["sender_balance_change"] = (
            "Change in sender's balance (newbalanceOrig - oldbalanceOrg). "
            "Negative = money sent, Positive = unusual."
        )

        # Receiver balance change
        self.X_engineered["receiver_balance_change"] = (
            self.X["newbalanceDest"] - self.X["oldbalanceDest"]
        )
        self.feature_descriptions["receiver_balance_change"] = (
            "Change in receiver's balance (newbalanceDest - oldbalanceDest). "
            "Usually positive for legitimate transfers."
        )

        # Absolute balance changes
        self.X_engineered["sender_balance_change_abs"] = np.abs(
            self.X_engineered["sender_balance_change"]
        )
        self.feature_descriptions["sender_balance_change_abs"] = (
            "Absolute value of sender's balance change."
        )

        self.X_engineered["receiver_balance_change_abs"] = np.abs(
            self.X_engineered["receiver_balance_change"]
        )
        self.feature_descriptions["receiver_balance_change_abs"] = (
            "Absolute value of receiver's balance change."
        )

        # Balance change ratio (relative to original balance, avoid division by zero)
        self.X_engineered["sender_balance_change_ratio"] = np.where(
            self.X["oldbalanceOrg"] != 0,
            self.X_engineered["sender_balance_change"] / np.abs(self.X["oldbalanceOrg"] + 1e-10),
            0
        )
        self.feature_descriptions["sender_balance_change_ratio"] = (
            "Sender's balance change relative to original balance. "
            "Large negative values indicate account depletion."
        )

        receiver_denom = np.abs(self.X["oldbalanceDest"] + 1e-10)
        self.X_engineered["receiver_balance_change_ratio"] = np.where(
            self.X["oldbalanceDest"] != 0,
            self.X_engineered["receiver_balance_change"] / receiver_denom,
            0
        )
        self.feature_descriptions["receiver_balance_change_ratio"] = (
            "Receiver's balance change relative to original balance."
        )

        print("✓ Created 6 balance-based features")

    def create_transaction_amount_features(self):
        """
        Create features based on transaction amounts.
        
        Rationale:
        - Very high transactions relative to account balance = suspicious
        - Transaction amount as % of available balance
        - Zero-balance accounts receiving money = money laundering pattern
        """
        print("\n[3/5] Creating transaction amount features...")

        # Amount as ratio of sender's original balance
        self.X_engineered["amount_ratio_sender_balance"] = np.where(
            self.X["oldbalanceOrg"] > 0,
            self.X["amount"] / (self.X["oldbalanceOrg"] + 1e-10),
            0
        )
        self.feature_descriptions["amount_ratio_sender_balance"] = (
            "Transaction amount relative to sender's original balance. "
            "Values > 1.0 mean transaction exceeds available balance (suspicious)."
        )

        # Amount as ratio of receiver's original balance
        self.X_engineered["amount_ratio_receiver_balance"] = np.where(
            self.X["oldbalanceDest"] > 0,
            self.X["amount"] / (self.X["oldbalanceDest"] + 1e-10),
            0
        )
        self.feature_descriptions["amount_ratio_receiver_balance"] = (
            "Transaction amount relative to receiver's original balance."
        )

        # Money completely depletes sender's account
        self.X_engineered["sender_depletes_account"] = (
            np.abs(self.X_engineered["sender_balance_change"]) >= 
            np.abs(self.X["oldbalanceOrg"]) * 0.99
        ).astype(int)
        self.feature_descriptions["sender_depletes_account"] = (
            "Binary: 1 if transaction depletes sender's account by ≥99%"
        )

        # Receiver starts with zero balance (common in money laundering)
        self.X_engineered["receiver_had_zero_balance"] = (
            self.X["oldbalanceDest"] == 0
        ).astype(int)
        self.feature_descriptions["receiver_had_zero_balance"] = (
            "Binary: 1 if receiver had zero balance before transaction"
        )

        # High-value transaction (top 10% by amount)
        amount_q90 = self.X["amount"].quantile(0.90)
        self.X_engineered["is_high_value_transaction"] = (
            self.X["amount"] >= amount_q90
        ).astype(int)
        self.feature_descriptions["is_high_value_transaction"] = (
            f"Binary: 1 if transaction amount ≥ 90th percentile ({amount_q90:.2f})"
        )

        print("✓ Created 5 transaction amount features")

    def create_account_type_features(self):
        """
        Create features based on account types and interactions.
        
        Rationale:
        - Certain account type combinations are more fraud-prone
        - Savings accounts have different risk profiles than current accounts
        """
        print("\n[4/5] Creating account type interaction features...")

        # Current to Savings (riskier transfer pattern)
        self.X_engineered["current_to_savings_transfer"] = (
            (self.X["acct_type_Current"] == 1) & 
            (self.X["type_TRANSFER"] == True)
        ).astype(int)
        self.feature_descriptions["current_to_savings_transfer"] = (
            "Binary: 1 if current account initiating TRANSFER (higher risk)"
        )

        # Cash out with unusual login (combined risk)
        self.X_engineered["cashout_with_unusual_login"] = (
            (self.X["type_CASH_OUT"] == True) & 
            (self.X["unusuallogin"] > 5)
        ).astype(int)
        self.feature_descriptions["cashout_with_unusual_login"] = (
            "Binary: 1 if CASH_OUT with high unusual login score"
        )

        # High branch fraud rate + high-value transaction
        self.X_engineered["high_fraud_branch_high_amount"] = (
            (self.X["branch_fraud_rate"] > 0.01) & 
            (self.X["amount"] > self.X["amount"].quantile(0.75))
        ).astype(int)
        self.feature_descriptions["high_fraud_branch_high_amount"] = (
            "Binary: 1 if high-fraud-rate branch + high-value transaction"
        )

        # Already flagged + unusual login
        self.X_engineered["flagged_and_unusual"] = (
            (self.X["isFlaggedFraud"] == 1) & 
            (self.X["unusuallogin"] > 3)
        ).astype(int)
        self.feature_descriptions["flagged_and_unusual"] = (
            "Binary: 1 if already flagged by system + unusual login"
        )

        print("✓ Created 4 account type interaction features")

    def create_derived_features(self):
        """
        Create additional derived features combining multiple signals.
        
        Rationale:
        - Most fraud signals are weak individually but strong combined
        - ML models can learn these complex patterns
        """
        print("\n[5/5] Creating derived signal features...")

        # Suspicion score: count of suspicious indicators
        self.X_engineered["suspicion_score"] = (
            self.X_engineered["sender_depletes_account"] +
            self.X_engineered["receiver_had_zero_balance"] +
            self.X_engineered["cashout_with_unusual_login"] +
            self.X_engineered["high_fraud_branch_high_amount"] +
            self.X_engineered["flagged_and_unusual"]
        )
        self.feature_descriptions["suspicion_score"] = (
            "Cumulative suspicion score: sum of 5 binary fraud indicators. "
            "Range: 0-5. Higher = more suspicious."
        )

        # Risk exposure: combined metric of balance risk + amount risk
        self.X_engineered["risk_exposure_sender"] = (
            np.abs(self.X_engineered["sender_balance_change_ratio"]) * 
            self.X_engineered["amount_ratio_sender_balance"]
        )
        self.feature_descriptions["risk_exposure_sender"] = (
            "Combined sender risk: balance change ratio × amount ratio. "
            "Captures both account depletion and unusual amounts."
        )

        # Total money flow (both directions)
        self.X_engineered["total_money_flow"] = (
            np.abs(self.X_engineered["sender_balance_change"]) +
            np.abs(self.X_engineered["receiver_balance_change"])
        )
        self.feature_descriptions["total_money_flow"] = (
            "Total absolute money movement in transaction"
        )

        print("✓ Created 3 derived signal features")

    def validate_engineered_features(self):
        """
        Validate generated features for quality and distribution.
        """
        print("\n✓ Feature Validation:")

        # Check NaN values
        nan_counts = self.X_engineered.isna().sum()
        if nan_counts.sum() > 0:
            print(f"  ⚠ WARNING: {nan_counts.sum()} NaN values found")
            print(nan_counts[nan_counts > 0])
        else:
            print(f"  ✓ No NaN values in {len(self.X_engineered.columns)} features")

        # Check infinite values
        inf_count = np.isinf(self.X_engineered.select_dtypes(include=[np.number])).sum().sum()
        if inf_count > 0:
            print(f"  ⚠ WARNING: {inf_count} infinite values found")
        else:
            print(f"  ✓ No infinite values")

        # Feature statistics
        numeric_features = self.X_engineered.select_dtypes(include=[np.number]).columns
        print(f"  ✓ Numeric features: {len(numeric_features)}")
        print(f"  ✓ Binary features: {len(self.X_engineered.select_dtypes(include=['bool']))} (one-hot encoded)")

    def save_outputs(self, output_dir: Path = None):
        """Save engineered features and documentation."""
        if output_dir is None:
            output_dir = repo_root / "data" / "engineered_features"

        print(f"\n✓ Saving outputs to {output_dir}...")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save combined features (original + engineered)
        X_combined = pd.concat([self.X, self.X_engineered], axis=1)
        X_path = output_dir / "X_engineered.csv"
        X_combined.to_csv(X_path, index=False)
        print(f"  ✓ Features saved: {X_path}")
        print(f"    Shape: {X_combined.shape}")

        # Save engineered features only (for reference)
        X_eng_only = output_dir / "X_engineered_only.csv"
        self.X_engineered.to_csv(X_eng_only, index=False)
        print(f"  ✓ Engineered features only: {X_eng_only}")

        # Save target
        y_path = output_dir / "y_engineered.csv"
        self.y.to_csv(y_path, index=False, header=True)
        print(f"  ✓ Target saved: {y_path}")

        # Save feature documentation
        doc_path = output_dir / "feature_documentation.txt"
        with open(doc_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("ENGINEERED FEATURES DOCUMENTATION\n")
            f.write("=" * 80 + "\n")
            f.write(f"\nTotal Features: {len(self.X_engineered.columns)}\n")
            f.write(f"Total Rows: {len(self.X_engineered)}\n\n")

            for i, (feature_name, description) in enumerate(self.feature_descriptions.items(), 1):
                f.write(f"\n{i}. {feature_name}\n")
                f.write(f"   Description: {description}\n")
                if feature_name in self.X_engineered.columns:
                    dtype = self.X_engineered[feature_name].dtype
                    f.write(f"   Data Type: {dtype}\n")
                    if dtype in ['int64', 'float64']:
                        stats = self.X_engineered[feature_name].describe()
                        f.write(f"   Min: {stats['min']:.4f}, Max: {stats['max']:.4f}, Mean: {stats['mean']:.4f}\n")

        print(f"  ✓ Documentation: {doc_path}")

        # Save list of all features
        all_features = list(self.X.columns) + list(self.X_engineered.columns)
        features_path = output_dir / "all_features.txt"
        with open(features_path, 'w') as f:
            for i, fname in enumerate(all_features, 1):
                f.write(f"{i:2d}. {fname}\n")
        print(f"  ✓ All features list: {features_path}")

        # Save metadata
        metadata_path = output_dir / "metadata.txt"
        with open(metadata_path, 'w') as f:
            f.write(f"total_rows: {len(self.X_engineered)}\n")
            f.write(f"original_features: {len(self.X.columns)}\n")
            f.write(f"engineered_features: {len(self.X_engineered.columns)}\n")
            f.write(f"total_features: {len(all_features)}\n")
            f.write(f"target_distribution: {self.y.value_counts().to_dict()}\n")
        print(f"  ✓ Metadata: {metadata_path}")

    def run(self, input_dir: Path = None, output_dir: Path = None):
        """Execute complete feature engineering pipeline."""
        print("\n" + "=" * 70)
        print("PHASE 4: FEATURE ENGINEERING PIPELINE")
        print("=" * 70)

        self.load_data(input_dir)
        self.X_engineered = pd.DataFrame(index=self.X.index)

        self.create_balance_features()
        self.create_transaction_amount_features()
        self.create_account_type_features()
        self.create_derived_features()
        self.validate_engineered_features()
        self.save_outputs(output_dir)

        print("\n" + "=" * 70)
        print("✓ FEATURE ENGINEERING COMPLETE")
        print("=" * 70)
        print(f"\nOriginal features: {len(self.X.columns)}")
        print(f"Engineered features: {len(self.X_engineered.columns)}")
        print(f"Total features: {len(self.X.columns) + len(self.X_engineered.columns)}")
        print(f"\nNext phase: Model Training")
        print()


if __name__ == "__main__":
    engineer = FeatureEngineer()
    engineer.run()
