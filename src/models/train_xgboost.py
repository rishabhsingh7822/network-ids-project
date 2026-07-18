import numpy as np
import joblib
import logging
import optuna
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BASE_DIR / 'data/processed'
MODELS_DIR = BASE_DIR / 'src/models'

def objective(trial, X_train, y_train, X_val, y_val):
    params = {
        'n_estimators':        trial.suggest_int('n_estimators', 50, 150),
        'max_depth':           trial.suggest_int('max_depth', 4, 8),
        'learning_rate':       trial.suggest_float('learning_rate', 0.05, 0.3),
        'subsample':           trial.suggest_float('subsample', 0.7, 1.0),
        'colsample_bytree':    trial.suggest_float('colsample_bytree', 0.7, 1.0),
        'eval_metric':         'mlogloss',
        'random_state':        42,
        'n_jobs':              -1,
        'tree_method':         'hist',
        'early_stopping_rounds': 10   # ← moved here in new XGBoost
    }
    model = XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    preds = model.predict(X_val)
    return f1_score(y_val, preds, average='macro')

def train_xgboost():
    logger.info("Loading scaled data...")
    X_train, X_test, y_train, y_test = joblib.load(
        PROCESSED_DIR / 'scaled_data.pkl'
    )

    # Use small validation set for Optuna speed
    from sklearn.model_selection import train_test_split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train,
        test_size=0.1,
        random_state=42,
        stratify=y_train
    )
    logger.info(f"Optuna search on {len(X_tr):,} samples...")

    # Optuna hyperparameter search — 5 trials
    study = optuna.create_study(direction='maximize')
    study.optimize(
        lambda trial: objective(trial, X_tr, y_tr, X_val, y_val),
        n_trials=5,
        show_progress_bar=True
    )

    logger.info(f"Best Macro F1: {study.best_value:.4f}")
    logger.info(f"Best params: {study.best_params}")

    # Train final model with best params on full training data
    logger.info("Training final XGBoost with best params...")
    best_params = study.best_params
    best_params.update({
        'eval_metric': 'mlogloss',
        'random_state': 42,
        'n_jobs': -1,
        'tree_method': 'hist'
    })

    xgb = XGBClassifier(**best_params)
    xgb.fit(X_train, y_train, verbose=False)

    # Evaluate
    y_pred = xgb.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')

    logger.info(f"Accuracy:  {acc:.4f}")
    logger.info(f"Macro F1:  {macro_f1:.4f}")

    le = joblib.load(PROCESSED_DIR / 'label_encoder.pkl')
    print("\n" + "="*60)
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Save model and best params
    joblib.dump(xgb, MODELS_DIR / 'xgboost.pkl')
    joblib.dump(study.best_params, MODELS_DIR / 'xgboost_best_params.pkl')
    logger.info("XGBoost saved to src/models/xgboost.pkl")

    return xgb, acc, macro_f1

if __name__ == '__main__':
    train_xgboost()