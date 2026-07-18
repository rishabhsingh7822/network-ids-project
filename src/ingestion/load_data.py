import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RAW_DIR = Path('data/raw')
PROCESSED_DIR = Path('data/processed')

def load_and_merge_csv_files() -> pd.DataFrame:
    dfs = []
    for f in sorted(RAW_DIR.glob('*.csv')):
        logger.info(f'Loading {f.name}...')
        df = pd.read_csv(f, low_memory=False)
        df.columns = df.columns.str.strip()
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f'Total records: {len(combined):,}')
    return combined

def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # Replace infinite values with NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    # Flag rows with negative values
    neg_mask = df.select_dtypes(include=np.number).lt(0).any(axis=1)
    logger.info(f'Rows with negative values: {neg_mask.sum()}')
    # Drop rows where label is NaN
    df = df.dropna(subset=['Label'])
    logger.info(f'Records after validation: {len(df):,}')
    return df

if __name__ == '__main__':
    df = load_and_merge_csv_files()
    df = validate_dataframe(df)
    out = PROCESSED_DIR / 'combined_clean.parquet'
    df.to_parquet(out, index=False)
    logger.info(f'Saved to {out}')