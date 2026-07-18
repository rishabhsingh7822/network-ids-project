import numpy as np
# pyrefly: ignore [missing-import]
import joblib
import logging
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BASE_DIR / 'data/processed'
MODELS_DIR = BASE_DIR / 'src/models'

def train_random_forest():
    logger.info("Loading scaled data...")
    X_train, X_test, y_train, y_test = joblib.load(
        PROCESSED_DIR / 'scaled_data.pkl'
    )

    logger.info(f"Training shape: {X_train.shape}")
    logger.info("Training Random Forest...")

    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=10,
        n_jobs=-1,        # use all CPU cores
        random_state=42,
        verbose=1
    )

    rf.fit(X_train, y_train)
    logger.info("Training complete!")

    # Evaluate
    logger.info("Evaluating on test set...")
    y_pred = rf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')

    logger.info(f"Accuracy:  {acc:.4f}")
    logger.info(f"Macro F1:  {macro_f1:.4f}")

    # Load label encoder for readable report
    le = joblib.load(PROCESSED_DIR / 'label_encoder.pkl')
    print("\n" + "="*60)
    print(classification_report(
        y_test, y_pred,
        target_names=le.classes_
    ))

    # Save model
    joblib.dump(rf, MODELS_DIR / 'random_forest.pkl')
    logger.info("Model saved to src/models/random_forest.pkl")

    return rf, acc, macro_f1

if __name__ == '__main__':
    train_random_forest()