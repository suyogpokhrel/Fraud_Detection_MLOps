from pathlib import Path
import os
import sys
import pandas as pd
from database.db_connection import get_connection

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

from src.cache.redis_client import save_dataframe, clear_pipeline_cache


def load_processed_csv(repo_root: Path):
    csv_path = repo_root / "data" / "processed" / "processed_data.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Processed CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    rename_map = {
        "Acct type": "acct_type",
        "Date of transaction": "date_of_transaction",
        "Time of day": "time_of_day",
    }
    df = df.rename(columns=rename_map)

    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

    insert_sql = """
        INSERT INTO processed_transactions (
            step, type, branch, amount, nameOrig, oldbalanceOrg, newbalanceOrig,
            nameDest, oldbalanceDest, newbalanceDest, unusuallogin, isFlaggedFraud,
            acct_type, date_of_transaction, time_of_day, isFraud, datetime
        ) VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
    """

    conn = get_connection()
    try:
        cur = conn.cursor()
        rows = []
        for _, row in df.iterrows():
            rows.append((
                int(row["step"]) if pd.notna(row["step"]) else None,
                row.get("type"),
                row.get("branch"),
                float(row["amount"]) if pd.notna(row["amount"]) else None,
                row.get("nameOrig"),
                float(row["oldbalanceOrg"]) if pd.notna(row["oldbalanceOrg"]) else None,
                float(row["newbalanceOrig"]) if pd.notna(row["newbalanceOrig"]) else None,
                row.get("nameDest"),
                float(row["oldbalanceDest"]) if pd.notna(row["oldbalanceDest"]) else None,
                float(row["newbalanceDest"]) if pd.notna(row["newbalanceDest"]) else None,
                int(row["unusuallogin"]) if pd.notna(row["unusuallogin"]) else None,
                int(row["isFlaggedFraud"]) if pd.notna(row["isFlaggedFraud"]) else None,
                str(row.get("acct_type")) if pd.notna(row.get("acct_type")) else None,
                str(row.get("date_of_transaction")) if pd.notna(row.get("date_of_transaction")) else None,
                str(row.get("time_of_day")) if pd.notna(row.get("time_of_day")) else None,
                int(row["isFraud"]) if pd.notna(row["isFraud"]) else None,
                str(row["datetime"]) if pd.notna(row["datetime"]) else None,
            ))
        cur.executemany(insert_sql, rows)
        conn.commit()
        print(f"Inserted {len(rows)} rows into processed_transactions")
    finally:
        conn.close()

    # Clear old pipeline cache, then cache fresh raw data for next steps
    clear_pipeline_cache()
    saved = save_dataframe("fraud:raw_data", df)
    if saved:
        print(f"Cached {len(df)} rows to Redis key 'fraud:raw_data'")
    else:
        print("Redis unavailable — pipeline will use MariaDB directly")


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[2]
    load_processed_csv(repo_root)
