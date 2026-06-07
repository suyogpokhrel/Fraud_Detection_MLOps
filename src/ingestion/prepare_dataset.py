from pathlib import Path
import pandas as pd


def prepare_dataset(repo_root: Path):
    raw_path = repo_root / "data" / "raw" / "Datasets.csv"
    processed_dir = repo_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / "processed_data.csv"

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw dataset not found: {raw_path}")

    df = pd.read_csv(raw_path)
    df = df.copy()

    # 1) Drop duplicated index column if present
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    # 2) Drop rows missing target and make target integer
    df = df.dropna(subset=['isFraud'])
    df['isFraud'] = df['isFraud'].astype(int)

    # 3) Fill small categorical/text missings
    if 'type' in df.columns:
        df['type'] = df['type'].fillna(df['type'].mode().iat[0])
    if 'Acct type' in df.columns:
        df['Acct type'] = df['Acct type'].fillna(df['Acct type'].mode().iat[0])
    if 'nameOrig' in df.columns:
        df['nameOrig'] = df['nameOrig'].fillna('UNKNOWN')
    if 'nameDest' in df.columns:
        df['nameDest'] = df['nameDest'].fillna('UNKNOWN')

    # 4) Drop rows with rare numeric missings (safe since counts are tiny)
    numeric_candidates = ['amount', 'oldbalanceOrg', 'oldbalanceDest', 'newbalanceDest', 'newbalanceOrig']
    for c in numeric_candidates:
        if c in df.columns and df[c].isnull().sum() > 0:
            df = df.dropna(subset=[c])

    # 5) Parse datetime
    # Observed date format is day/month/year (e.g. 13/1/2018). Time of day is categorical
    # like 'Morning', 'Afternoon', 'Night'. We map those to representative times.
    if 'Date of transaction' in df.columns and 'Time of day' in df.columns:
        df['Date of transaction'] = df['Date of transaction'].astype(str)
        # parse date with explicit day-first format
        df['date_parsed'] = pd.to_datetime(df['Date of transaction'], format='%d/%m/%Y', errors='coerce')
        df['Time of day'] = df['Time of day'].astype(str)
        tod_map = {
            'Morning': '09:00:00',
            'Afternoon': '15:00:00',
            'Night': '21:00:00',
            'Evening': '18:00:00',
            'Midnight': '00:00:00'
        }
        df['tod_time'] = df['Time of day'].map(tod_map).fillna('12:00:00')
        df['datetime'] = pd.to_datetime(df['date_parsed'].dt.strftime('%Y-%m-%d') + ' ' + df['tod_time'], errors='coerce')
        df = df.drop(columns=['date_parsed','tod_time'])
    else:
        df['datetime'] = pd.NaT

    # 5b) Drop rows where we could not construct a valid datetime.
    # These rows have missing transaction dates and are not usable for time-aware storage.
    df = df.dropna(subset=['datetime'])

    # 6) Set categorical dtypes
    for c in ['type', 'branch', 'Acct type']:
        if c in df.columns:
            df[c] = df[c].astype('category')

    # 7) Summary prints
    print(f"shape after cleaning: {df.shape}")
    dt_failures = int(df['datetime'].isnull().sum()) if 'datetime' in df.columns else 0
    print(f"datetime parse failures: {dt_failures}")
    dups = int(df.duplicated().sum())
    print(f"duplicates: {dups}")
    print('\nTarget counts:')
    print(df['isFraud'].value_counts())
    print('\nTarget percent:')
    print((df['isFraud'].value_counts(normalize=True) * 100).round(3))

    # 8) Save processed
    df.to_csv(out_path, index=False)
    print(f"\nSaved processed dataset to: {out_path}")


if __name__ == '__main__':
    repo_root = Path(__file__).resolve().parents[2]
    prepare_dataset(repo_root)
