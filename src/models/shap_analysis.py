import shap
import joblib
import numpy as np
import matplotlib.pyplot as plt
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BASE_DIR / 'data/processed'
MODELS_DIR    = BASE_DIR / 'src/models'
DOCS_DIR      = BASE_DIR / 'docs'

def run_shap_analysis():
    DOCS_DIR.mkdir(exist_ok=True)
    logger.info("Loading XGBoost model and data...")
    xgb = joblib.load(MODELS_DIR / 'xgboost.pkl')

    X_train, X_test, y_train, y_test = joblib.load(
        PROCESSED_DIR / 'scaled_data.pkl'
    )

    # Use small sample for SHAP — 500 rows is enough
    sample_idx = np.random.choice(len(X_test), 500, replace=False)
    X_sample = X_test[sample_idx]

    logger.info("Computing SHAP values (this takes 2-3 mins)...")
    explainer   = shap.TreeExplainer(xgb)
    shap_values = explainer.shap_values(X_sample)

    # Get feature names
    pipeline    = joblib.load(PROCESSED_DIR / 'pipeline.pkl')
    import pandas as pd
    feature_names = joblib.load(PROCESSED_DIR / 'train_test_split.pkl')[0].columns.tolist()

    # Plot 1 — Summary plot (top 20 features)
    logger.info("Generating SHAP summary plot...")
    plt.figure()
    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=feature_names,
        max_display=20,
        show=False
    )
    plt.tight_layout()
    plt.savefig(DOCS_DIR / 'shap_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved docs/shap_summary.png")

    # Plot 2 — Bar plot (mean absolute SHAP values)
    plt.figure()
    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=feature_names,
        plot_type='bar',
        max_display=20,
        show=False
    )
    plt.tight_layout()
    plt.savefig(DOCS_DIR / 'shap_bar.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved docs/shap_bar.png")

    # Print top 10 most important features
    shap_array = np.array(shap_values)
    if shap_array.ndim == 3:
        # multiclass: shape (n_classes, n_samples, n_features)
        mean_shap = np.abs(shap_array).mean(axis=(0, 1))
    elif shap_array.ndim == 4:
        # some versions: shape (n_samples, n_classes, n_features, ...)
        mean_shap = np.abs(shap_array).mean(axis=(0, 1, 2))
    else:
        mean_shap = np.abs(shap_array).mean(axis=0)

    feature_importance = sorted(
        zip(feature_names, mean_shap),
        key=lambda x: x[1],
        reverse=True
    )[:10]

    print("\n TOP 10 MOST IMPORTANT FEATURES:")
    print("="*40)
    for i, (feat, score) in enumerate(feature_importance, 1):
        print(f"{i:2}. {feat:<35} {score:.4f}")

    logger.info("SHAP analysis complete!")

if __name__ == '__main__':
    run_shap_analysis()