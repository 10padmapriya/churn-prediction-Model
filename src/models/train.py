"""
src/models/train.py

Train multiple models, optimize decision threshold for business cost matrix,
evaluate with full metrics, and save the best model + preprocessor.
"""

import logging
from pathlib import Path

import joblib
import numpy as np
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)


# ─────────────────────────────────────────────
# Cost-sensitive threshold optimization
# ─────────────────────────────────────────────


def optimize_threshold(
    y_true, y_prob, cost_fn: float, cost_fp: float
) -> tuple[float, float]:
    """
    Find the decision threshold that minimizes total business cost.

    Total cost = (FN × cost_FN) + (FP × cost_FP)

    This is the key insight that separates ML engineers from data analysts:
    default threshold of 0.5 optimizes accuracy, not business outcomes.
    """
    thresholds = np.linspace(0.1, 0.9, 80)
    costs = []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        total_cost = (fn * cost_fn) + (fp * cost_fp)
        costs.append(total_cost)

    best_idx = np.argmin(costs)
    best_threshold = thresholds[best_idx]
    best_cost = costs[best_idx]

    logger.info(
        f"Optimal threshold: {best_threshold:.3f} | Min cost: ${best_cost:,.0f}"
    )
    return best_threshold, best_cost


# ─────────────────────────────────────────────
# Model definitions
# ─────────────────────────────────────────────


def get_models(config: dict) -> dict:
    cfg = config["models"]
    return {
        "LogisticRegression": LogisticRegression(**cfg["logistic_regression"]),
        "XGBoost": XGBClassifier(**cfg["xgboost"], verbosity=0),
        "LightGBM": LGBMClassifier(**cfg["lightgbm"], verbose=-1),
    }


# ─────────────────────────────────────────────
# Full evaluation report
# ─────────────────────────────────────────────


def evaluate_model(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    config: dict,
    threshold: float = None,
) -> dict:
    """Returns a full dict of metrics for one model."""
    y_prob = model.predict_proba(X_test)[:, 1]

    cost_fn = config["business"]["cost_false_negative"]
    cost_fp = config["business"]["cost_false_positive"]

    if threshold is None:
        threshold, _ = optimize_threshold(y_test, y_prob, cost_fn, cost_fp)

    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    # Business impact calculations
    actual_churners = y_test.sum()
    caught_churners = tp
    missed_churners = fn
    false_alarms = fp

    avg_clv = (
        config["business"]["avg_monthly_revenue"] * config["business"]["avg_clv_months"]
    )
    revenue_saved = caught_churners * avg_clv
    retention_cost = (caught_churners + false_alarms) * config["business"][
        "cost_false_positive"
    ]
    net_value = revenue_saved - retention_cost

    return {
        # Standard ML metrics
        "roc_auc": roc_auc_score(y_test, y_prob),
        "avg_precision": average_precision_score(y_test, y_prob),
        "threshold": threshold,
        "precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
        "recall": tp / (tp + fn) if (tp + fn) > 0 else 0,
        "f1": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
        # Confusion matrix
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        # Business impact
        "actual_churners": int(actual_churners),
        "caught_churners": int(caught_churners),
        "missed_churners": int(missed_churners),
        "false_alarms": int(false_alarms),
        "revenue_saved_usd": float(revenue_saved),
        "retention_cost_usd": float(retention_cost),
        "net_value_usd": float(net_value),
        # Curves (for dashboard plotting)
        "roc_curve": roc_curve(y_test, y_prob),
        "pr_curve": precision_recall_curve(y_test, y_prob),
    }


# ─────────────────────────────────────────────
# Training pipeline
# ─────────────────────────────────────────────


def train_all_models(
    X_train,
    y_train,
    X_test,
    y_test,
    config: dict,
    feature_names: list[str],
) -> tuple[object, dict, dict]:
    """
    Train all models, compare CV AUC, return best model + all results.
    """
    models = get_models(config)
    results = {}
    cv = StratifiedKFold(
        n_splits=config["training"]["cv_folds"], shuffle=True, random_state=42
    )

    # SMOTE: only applied to training data, never test
    if config["training"]["use_smote"]:
        logger.info("Applying SMOTE to training set...")
        smote = SMOTE(
            sampling_strategy=config["training"]["smote_sampling_strategy"],
            random_state=42,
        )
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        logger.info(f"SMOTE: {y_train.sum()} → {y_train_res.sum()} minority samples")
    else:
        X_train_res, y_train_res = X_train, y_train

    best_model_name = None
    best_auc = -1

    for name, model in models.items():
        logger.info(f"\nTraining {name}...")

        # Cross-validated AUC on training set
        cv_scores = cross_val_score(
            model,
            X_train_res,
            y_train_res,
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1,
        )
        logger.info(f"  CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        # Final fit on full resampled training set
        model.fit(X_train_res, y_train_res)

        # Evaluate on held-out test set (no SMOTE applied here)
        metrics = evaluate_model(
            model, X_train_res, y_train_res, X_test, y_test, config
        )
        metrics["cv_auc_mean"] = cv_scores.mean()
        metrics["cv_auc_std"] = cv_scores.std()
        results[name] = metrics

        logger.info(
            f"  Test AUC: {metrics['roc_auc']:.4f} | "
            f"Recall: {metrics['recall']:.3f} | "
            f"Precision: {metrics['precision']:.3f} | "
            f"Net value: ${metrics['net_value_usd']:,.0f}"
        )

        if metrics["roc_auc"] > best_auc:
            best_auc = metrics["roc_auc"]
            best_model_name = name
            best_model = model

    logger.info(f"\n🏆 Best model: {best_model_name} (AUC={best_auc:.4f})")
    return best_model, best_model_name, results


def save_artifacts(model, preprocessor, feature_names, config, engineer=None):
    """Save model + preprocessor + engineer for the API to load."""
    out = Path("data/processed")
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / "best_model.joblib")
    joblib.dump(preprocessor, out / "preprocessor.joblib")
    joblib.dump(feature_names, out / "feature_names.joblib")
    if engineer is not None:
        joblib.dump(engineer, out / "engineer.joblib")
        logger.info(
            f"Engineer persisted (monthly_charges_75th_={engineer.monthly_charges_75th_:.2f})"
        )
    logger.info("Artifacts saved → data/processed/")
