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

    # Drop unnamed index if present
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    # Drop rows missing target
    df = df.dropna(subset=['isFraud'])
    df['isFraud'] = df['isFraud'].astype(int)

    # Fill missing categoricals
    if 'type' in df.columns:
        df['type'] = df['type'].fillna(df['type'].mode().iat[0])
    if 'nameOrig' in df.columns:
        df['nameOrig'] = df['nameOrig'].fillna('UNKNOWN')
    if 'nameDest' in df.columns:
        df['nameDest'] = df['nameDest'].fillna('UNKNOWN')

    # Fill missing numerics
    for col in ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']:
        if col in df.columns and df[col].isnull().sum() > 0:
            df = df.dropna(subset=[col])

    # Set categorical dtypes
    if 'type' in df.columns:
        df['type'] = df['type'].astype('category')

    # Summary
    print(f"Shape after cleaning: {df.shape}")
    print(f"Duplicates: {df.duplicated().sum()}")
    print(f"\nTarget counts:")
    print(df['isFraud'].value_counts())
    print(f"\nFraud rate: {df['isFraud'].mean()*100:.2f}%")

    df.to_csv(out_path, index=False)
    print(f"\nSaved processed dataset to: {out_path}")


if __name__ == '__main__':
    repo_root = Path(__file__).resolve().parents[2]
    prepare_dataset(repo_root)
