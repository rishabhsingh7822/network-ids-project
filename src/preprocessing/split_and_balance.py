# src/preprocessing/split_and_balance.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from collections import Counter
from pathlib import Path
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BASE_DIR / 'data/processed'

def prepare_data():
    logger.info("Loading clean parquet file...")
    df = pd.read_parquet(PROCESSED_DIR / 'combined_clean.parquet')

    # Drop highly correlated features
    import json
    with open(PROCESSED_DIR / 'features_to_drop.json') as f:
        features_to_drop = json.load(f)
    df = df.drop(columns=[c for c in features_to_drop if c in df.columns])

    # Encode labels
    le = LabelEncoder()
    df['label_encoded'] = le.fit_transform(df['Label'])
    joblib.dump(le, PROCESSED_DIR / 'label_encoder.pkl')

    # Features and target
    X = df.drop(columns=['Label', 'label_encoded'])
    y = df['label_encoded']

    # STEP 1: Split FIRST
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

    # STEP 2: Impute NaN values
    logger.info("Imputing missing values...")
    imputer = SimpleImputer(strategy='median')
    X_train = pd.DataFrame(
        imputer.fit_transform(X_train),
        columns=X_train.columns
    )
    X_test = pd.DataFrame(
        imputer.transform(X_test),
        columns=X_test.columns
    )
    joblib.dump(imputer, PROCESSED_DIR / 'imputer.pkl')

    # STEP 3: SMOTE with small cap — max 2000 per minority class
    logger.info("Applying SMOTE (capped at 2000 per minority class)...")
    class_counts = Counter(y_train.tolist())
    sampling_strategy = {}
    for cls, count in class_counts.items():
        if count < 2000:
            sampling_strategy[cls] = 2000

    logger.info(f"Classes to oversample: {len(sampling_strategy)}")

    smote = SMOTE(
        random_state=42,
        k_neighbors=3,
        sampling_strategy=sampling_strategy
    )
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    logger.info(f"Training size after SMOTE: {len(X_train_bal):,}")

    # Save
    joblib.dump((X_train_bal, X_test, y_train_bal, y_test),
                PROCESSED_DIR / 'train_test_split.pkl')
    logger.info("Saved! Week 2 Day 1-2 complete.")

    return X_train_bal, X_test, y_train_bal, y_test

if __name__ == '__main__':
    prepare_data()