# src/preprocessing/feature_engineer.py
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def add_entropy_features(df: pd.DataFrame) -> pd.DataFrame:
    # Packet size entropy — high entropy = DDoS signal
    pkt_cols = [c for c in df.columns if 'Length' in c or 'length' in c]
    if pkt_cols:
        vals = df[pkt_cols].values.astype(float)
        row_sums = vals.sum(axis=1, keepdims=True) + 1e-9
        probs = vals / row_sums
        df['pkt_size_entropy'] = -np.sum(
            probs * np.log2(probs + 1e-9), axis=1
        )
        logger.info("Added pkt_size_entropy feature")
    return df

def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    # SYN/total packets ratio — port scan signal
    if 'SYN Flag Count' in df.columns and 'Total Fwd Packets' in df.columns:
        total = df['Total Fwd Packets'] + df['Total Backward Packets'] + 1
        df['syn_ratio'] = df['SYN Flag Count'] / total
        logger.info("Added syn_ratio feature")

    # Bytes sent/received ratio — exfiltration signal
    if 'Total Length of Fwd Packets' in df.columns and 'Bwd Packet Length Max' in df.columns:
        bwd = df['Bwd Packet Length Max'] + 1
        df['bytes_ratio'] = df['Total Length of Fwd Packets'] / bwd
        logger.info("Added bytes_ratio feature")

    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info(f"Features before engineering: {df.shape[1]}")
    df = add_entropy_features(df)
    df = add_ratio_features(df)
    logger.info(f"Features after engineering: {df.shape[1]}")
    return df

if __name__ == '__main__':
    import joblib
    from pathlib import Path

    BASE_DIR = Path(__file__).resolve().parents[2]
    PROCESSED_DIR = BASE_DIR / 'data/processed'

    # Load training data
    X_train, X_test, y_train, y_test = joblib.load(
        PROCESSED_DIR / 'train_test_split.pkl'
    )

    # Engineer features
    X_train = engineer_features(X_train)
    X_test = engineer_features(X_test)

    # Save updated splits
    joblib.dump((X_train, X_test, y_train, y_test),
                PROCESSED_DIR / 'train_test_split.pkl')
    logger.info("Feature engineering complete!")