"""
src/models/explain.py

SHAP-based model explainability.
Produces both global feature importance and per-customer explanations.
This is what turns a black-box model into an actionable business tool.
"""

import numpy as np
import pandas as pd
import shap
import logging
import warnings

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


def get_explainer(model, X_background: np.ndarray):
    """
    Build a SHAP explainer appropriate for the model type.
    - Tree models (XGBoost, LightGBM): fast TreeExplainer
    - Others: KernelExplainer (slower, model-agnostic)
    """
    model_type = type(model).__name__

    if model_type in ("XGBClassifier", "LGBMClassifier", "RandomForestClassifier"):
        logger.info(f"Using TreeExplainer for {model_type}")
        explainer = shap.TreeExplainer(model)
    else:
        logger.info(f"Using LinearExplainer for {model_type}")
        explainer = shap.LinearExplainer(model, X_background)

    return explainer


def compute_shap_values(
    explainer,
    X: np.ndarray,
    feature_names: list[str],
    max_samples: int = 500,
) -> pd.DataFrame:
    """
    Compute SHAP values and return as a labeled DataFrame.
    Caps at max_samples to keep it fast for dashboards.
    """
    X_sample = X[:max_samples]
    shap_values = explainer.shap_values(X_sample)

    # Binary classifiers may return list [neg_class, pos_class]
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    df_shap = pd.DataFrame(shap_values, columns=feature_names)
    return df_shap


def global_feature_importance(shap_df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """
    Mean absolute SHAP value per feature — the 'global' importance.
    More trustworthy than model.feature_importances_ because it accounts
    for direction and interaction effects.
    """
    importance = shap_df.abs().mean().sort_values(ascending=False)
    return importance.head(top_n).reset_index().rename(
        columns={"index": "feature", 0: "mean_abs_shap"}
    )


def explain_single_customer(
    explainer,
    X_customer: np.ndarray,
    feature_names: list[str],
    baseline_prob: float,
) -> dict:
    """
    Returns a waterfall-style explanation for ONE customer.
    Used by the API /explain endpoint to power the dashboard.

    Returns:
        {
          "base_value": float,         # avg model output
          "prediction_prob": float,    # this customer's churn probability
          "features": [
            {"feature": str, "value": float, "shap_value": float},
            ...
          ]
        }
    """
    shap_vals = explainer.shap_values(X_customer.reshape(1, -1))
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]
    shap_vals = shap_vals[0]

    features = [
        {
            "feature": name,
            "raw_shap": float(val),
            "abs_shap": float(abs(val)),
            "direction": "increases_churn" if val > 0 else "decreases_churn",
        }
        for name, val in zip(feature_names, shap_vals)
    ]

    # Sort by absolute impact
    features.sort(key=lambda x: x["abs_shap"], reverse=True)

    base_value = float(explainer.expected_value)
    if isinstance(base_value, (list, np.ndarray)):
        base_value = float(base_value[1])

    return {
        "base_value": base_value,
        "prediction_prob": float(baseline_prob),
        "top_drivers": features[:10],  # top 10 for the waterfall chart
    }
