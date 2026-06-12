
from pathlib import Path
import sys
import pandas as pd

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

from src.cache.redis_client import load_dataframe, save_dataframe


def get_db_connection():
    try:
        import mariadb
        conn = mariadb.connect(
            host="127.0.0.1",
            port=3306,
            user="fraud_mlops_user",
            password="fraud_mlops_pass",
            database="fraud_mlops"
        )
        return conn
    except Exception as e:
        print(f"  [MariaDB] Connection failed: {e}")
        return None


def load_processed_csv(repo_root: Path):
    processed_path = repo_root / "data" / "processed" / "processed_data.csv"

    print("\n" + "=" * 60)
    print("PHASE 2: LOAD TO MARIADB")
    print("=" * 60)

    # Try Redis first
    print("\n[1/3] Loading data...")
    df = load_dataframe("pipeline:processed")
    if df is not None:
        print(f"  Loaded from Redis cache: {df.shape}")
    else:
        print(f"  Redis unavailable, loading from CSV...")
        if not processed_path.exists():
            raise FileNotFoundError(f"Processed CSV not found: {processed_path}")
        df = pd.read_csv(processed_path)
        print(f"  Loaded from CSV: {df.shape}")
        save_dataframe("pipeline:processed", df)

    # Keep only columns that match the schema
    cols = ["step", "type", "amount", "nameOrig",
            "oldbalanceOrg", "newbalanceOrig",
            "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud"]
    df = df[[c for c in cols if c in df.columns]]
    print(f"  Columns: {list(df.columns)}")

    # Connect to MariaDB
    print("\n[2/3] Connecting to MariaDB...")
    conn = get_db_connection()
    if conn is None:
        print("  MariaDB unavailable — skipping DB load.")
        print("  Data is cached in Redis for downstream steps.")
        return

    cursor = conn.cursor()
    cursor.execute("TRUNCATE TABLE processed_transactions")

    # Insert rows
    print("\n[3/3] Inserting rows...")
    insert_sql = """
        INSERT INTO processed_transactions
        (step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig,
         nameDest, oldbalanceDest, newbalanceDest, isFraud)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    batch_size = 5000
    total = 0
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i + batch_size]
        rows = [tuple(row) for row in batch[cols[:-1] + ["isFraud"]].itertuples(index=False)]
        cursor.executemany(insert_sql, rows)
        conn.commit()
        total += len(batch)
        print(f"  Inserted {total:,} / {len(df):,} rows...", end="\r")

    print(f"\n  Total rows inserted: {total:,}")

    # Verify
    cursor.execute("SELECT COUNT(*) FROM processed_transactions")
    count = cursor.fetchone()[0]
    fraud_count = df["isFraud"].sum()
    print(f"  DB row count:  {count:,}")
    print(f"  Fraud rows:    {fraud_count:,}")

    cursor.close()
    conn.close()
    print("\n✓ LOAD TO MARIADB COMPLETE")


if __name__ == "__main__":
    load_processed_csv(repo_root)
