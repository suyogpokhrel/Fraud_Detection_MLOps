
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))


def run_drift_report(
    reference_path: Path = None,
    current_path: Path = None,
    output_dir: Path = None
):
    if reference_path is None:
        reference_path = repo_root / "data" / "preprocessed" / "X_preprocessed.csv"
    if current_path is None:
        current_path = repo_root / "data" / "engineered_features" / "X_engineered.csv"
    if output_dir is None:
        output_dir = repo_root / "reports" / "drift"

    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("EVIDENTLY AI - DATA DRIFT MONITORING")
    print("="*70)

    print("\n[1/3] Loading reference and current data...")
    reference_df = pd.read_csv(reference_path)
    current_df = pd.read_csv(current_path)

    common_cols = list(set(reference_df.columns) & set(current_df.columns))
    reference_df = reference_df[common_cols]
    current_df = current_df[common_cols]

    ref_sample = reference_df.sample(min(5000, len(reference_df)), random_state=42)
    cur_sample = current_df.sample(min(5000, len(current_df)), random_state=42)

    print(f"  ✓ Reference: {len(ref_sample)} rows, {len(common_cols)} features")
    print(f"  ✓ Current:   {len(cur_sample)} rows, {len(common_cols)} features")

    print("\n[2/3] Running drift analysis...")

    from evidently.legacy.report import Report
    from evidently.legacy.metric_preset import DataDriftPreset, DataQualityPreset

    report = Report(metrics=[
        DataDriftPreset(),
        DataQualityPreset(),
    ])
    report.run(reference_data=ref_sample, current_data=cur_sample)

    print("\n[3/3] Saving HTML report...")
    report_path = output_dir / "drift_report.html"
    report.save_html(str(report_path))
    print(f"  ✓ Report saved to: {report_path}")
    print("\n  Open in browser to view drift analysis!")
    print("✓ DRIFT MONITORING COMPLETE")

    return str(report_path)


if __name__ == "__main__":
    run_drift_report()
