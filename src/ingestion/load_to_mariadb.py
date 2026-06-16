from pathlib import Path
import sys
import os
import pandas as pd

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

from src.cache.redis_client import load_dataframe, save_dataframe


def get_db_connection():
    import mariadb
    host = os.getenv("DB_HOST", "172.18.0.1")
    try:
        conn = mariadb.connect(
            host=host, port=3306,
            user="fraud_user", password="fraud_password",
            database="fraud_detection", connect_timeout=5
        )
        print(f"  Connected to MariaDB at {host}")
        return conn
    except Exception as e:
        print(f"  [MariaDB] Connection failed at {host}: {e}")
        return None


def load_processed_csv(repo_root: Path):
    processed_path = repo_root / "data" / "processed" / "processed_data.csv"

    print("\n" + "=" * 60)
    print("PHASE 2: LOAD TO MARIADB")
    print("=" * 60)

    print("\n[1/3] Loading data...")
    df = load_dataframe("pipeline:processed")
    if df is not None:
        print(f"  Loaded from Redis cache: {df.shape}")
    else:
        print("  Redis unavailable, loading from CSV...")
        if not processed_path.exists():
            raise FileNotFoundError(f"Processed CSV not found: {processed_path}")
        df = pd.read_csv(processed_path)
        print(f"  Loaded from CSV: {df.shape}")
        save_dataframe("pipeline:processed", df)

    cols = ["step", "type", "amount", "nameOrig",
            "oldbalanceOrg", "newbalanceOrig",
            "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud"]
    df = df[[c for c in cols if c in df.columns]]
    print(f"  Columns: {list(df.columns)}")

    print("\n[2/3] Connecting to MariaDB...")
    conn = get_db_connection()
    if conn is None:
        print("  MariaDB unavailable - skipping DB load.")
        print("  Data is cached in Redis for downstream steps.")
        return

    print("\n[3/3] Inserting rows...")
    cursor = conn.cursor()
    cursor.execute("DELETE FROM processed_transactions")
    conn.commit()

    insert_sql = """INSERT INTO processed_transactions
        (step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig,
         nameDest, oldbalanceDest, newbalanceDest, isFraud)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""

    batch_size = 5000
    total = 0
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i + batch_size]
        rows = [tuple(row) for row in batch.itertuples(index=False)]
        cursor.executemany(insert_sql, rows)
        conn.commit()
        total += len(batch)
        print(f"  Inserted {total:,} / {len(df):,} rows...")

    cursor.execute("SELECT COUNT(*) FROM processed_transactions")
    count = cursor.fetchone()[0]
    cursor.close()
    conn.close()

    print(f"  DB row count: {count:,}")
    print(f"  Fraud rows:   {int(df[chr(34)+'isFraud'+chr(34)].sum()):,}")
    print("\n  NOTE: Native ColumnStore loaded separately via cpimport.")
    print("\n\u2713 LOAD TO MARIADB COMPLETE")


if __name__ == "__main__":
    load_processed_csv(repo_root)
