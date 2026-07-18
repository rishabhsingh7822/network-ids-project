import numpy as np
import joblib
import logging
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, classification_report

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BASE_DIR / 'data/processed'
MODELS_DIR = BASE_DIR / 'src/models'

def ensemble_predict(X, rf, xgb, weights=(0.4, 0.6)):
    # Get probabilities from each model
    rf_proba  = rf.predict_proba(X)   # shape: (n, 15)
    xgb_proba = xgb.predict_proba(X)  # shape: (n, 15)

    # Weighted average — XGBoost gets higher weight (better Macro F1)
    combined = weights[0] * rf_proba + weights[1] * xgb_proba

    # Pick class with highest combined probability
    return np.argmax(combined, axis=1)

def evaluate_ensemble():
    logger.info("Loading models and data...")
    rf  = joblib.load(MODELS_DIR / 'random_forest.pkl')
    xgb = joblib.load(MODELS_DIR / 'xgboost.pkl')
    X_train, X_test, y_train, y_test = joblib.load(
        PROCESSED_DIR / 'scaled_data.pkl'
    )
    le = joblib.load(PROCESSED_DIR / 'label_encoder.pkl')

    logger.info("Running ensemble predictions...")
    y_pred = ensemble_predict(X_test, rf, xgb)

    acc      = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')

    logger.info(f"Accuracy:  {acc:.4f}")
    logger.info(f"Macro F1:  {macro_f1:.4f}")

    print("\n" + "="*60)
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Save ensemble function parameters
    joblib.dump({'weights': (0.4, 0.6)}, MODELS_DIR / 'ensemble_config.pkl')
    logger.info("Ensemble config saved!")

    return acc, macro_f1

if __name__ == '__main__':
    evaluate_ensemble()