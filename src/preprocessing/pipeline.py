# src/preprocessing/pipeline.py
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from pathlib import Path
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BASE_DIR / 'data/processed'

def build_pipeline():
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

def fit_and_save():
    logger.info("Loading train/test split...")
    X_train, X_test, y_train, y_test = joblib.load(
        PROCESSED_DIR / 'train_test_split.pkl'
    )

    logger.info("Fitting pipeline on training data only...")
    pipe = build_pipeline()

    # Fit ONLY on training data
    X_train_scaled = pipe.fit_transform(X_train)
    # Transform test data using fitted pipeline
    X_test_scaled = pipe.transform(X_test)

    logger.info(f"X_train_scaled shape: {X_train_scaled.shape}")
    logger.info(f"X_test_scaled shape: {X_test_scaled.shape}")

    # Save pipeline and scaled data
    joblib.dump(pipe, PROCESSED_DIR / 'pipeline.pkl')
    joblib.dump(
        (X_train_scaled, X_test_scaled, y_train, y_test),
        PROCESSED_DIR / 'scaled_data.pkl'
    )
    logger.info("Pipeline saved to data/processed/pipeline.pkl")
    logger.info("Week 2 Day 5-6 complete!")

if __name__ == '__main__':
    fit_and_save()