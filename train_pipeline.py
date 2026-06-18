"""
train_pipeline.py

Master script: runs the full ML pipeline end-to-end.
Usage:
    python train_pipeline.py

Outputs (saved to data/processed/):
    - best_model.joblib
    - preprocessor.joblib
    - feature_names.joblib
    - eval_metrics.json
"""

import json
import yaml
import logging
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

from src.data.ingestion import run_ingestion, load_config
from src.features.engineering import prepare_features
from src.models.train import train_all_models, save_artifacts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 60)
    logger.info("CHURN PREDICTION PIPELINE — START")
    logger.info("=" * 60)

    # ── 1. Load config ───────────────────────────────────────────
    config = load_config("configs/config.yaml")

    # ── 2. Ingest & clean raw data ───────────────────────────────
    logger.info("\n[1/5] DATA INGESTION")
    df = run_ingestion("configs/config.yaml")

    # ── 3. Train/test split (stratified) ────────────────────────
    logger.info("\n[2/5] TRAIN/TEST SPLIT")
    from sklearn.model_selection import train_test_split
    df_train, df_test = train_test_split(
        df,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
        stratify=df[config["data"]["target_column"]],
    )
    logger.info(f"Train: {len(df_train):,} | Test: {len(df_test):,}")

    # ── 4. Feature engineering ───────────────────────────────────
    logger.info("\n[3/5] FEATURE ENGINEERING")
    X_train, y_train, preprocessor, feature_names, engineer = prepare_features(
        df_train, config, fit=True
    )
    X_test, y_test, _, _, _ = prepare_features(
        df_test, config, fit=False, preprocessor=preprocessor, engineer=engineer
    )

    # ── 5. Train & evaluate all models ──────────────────────────
    logger.info("\n[4/5] MODEL TRAINING")
    best_model, best_name, all_results = train_all_models(
        X_train, y_train,
        X_test, y_test,
        config,
        feature_names,
    )

    # ── 6. Save artifacts ────────────────────────────────────────
    logger.info("\n[5/5] SAVING ARTIFACTS")
    save_artifacts(best_model, preprocessor, feature_names, config, engineer=engineer)

    # ── Write optimal threshold back to config ────────────────────
    import yaml as _yaml
    best_threshold = all_results[best_name]["threshold"]
    config["training"]["default_threshold"] = round(float(best_threshold), 6)
    with open("configs/config.yaml", "w") as _f:
        _yaml.dump(config, _f, default_flow_style=False, sort_keys=False)
    logger.info(f"Optimal threshold {best_threshold:.4f} written to configs/config.yaml")

    # Save metrics (strip non-serializable curve data)
    serializable_results = {}
    for model_name, metrics in all_results.items():
        clean = {k: v for k, v in metrics.items() if k not in ("roc_curve", "pr_curve")}
        serializable_results[model_name] = clean

    metrics_out = Path("data/processed/eval_metrics.json")
    with open(metrics_out, "w") as f:
        json.dump(
            {"best_model": best_name, "models": serializable_results},
            f, indent=2
        )
    logger.info(f"Metrics saved → {metrics_out}")

    # ── Print summary ────────────────────────────────────────────
    best = all_results[best_name]
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE — RESULTS SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Best Model       : {best_name}")
    logger.info(f"ROC-AUC          : {best['roc_auc']:.4f}")
    logger.info(f"Recall           : {best['recall']:.4f}")
    logger.info(f"Precision        : {best['precision']:.4f}")
    logger.info(f"Decision Threshold: {best['threshold']:.3f}")
    logger.info(f"Churners caught  : {best['caught_churners']} / {best['actual_churners']}")
    logger.info(f"Revenue saved    : ${best['revenue_saved_usd']:>10,.0f}")
    logger.info(f"Retention cost   : ${best['retention_cost_usd']:>10,.0f}")
    logger.info(f"Net value        : ${best['net_value_usd']:>10,.0f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
