"""
src/models/monitor.py

Production monitoring utilities:
- Population Stability Index (PSI) for feature drift detection
- Prediction distribution monitoring
- Automated retraining trigger logic

Usage:
    python src/models/monitor.py --reference data/processed/telco_processed.csv \
                                  --current data/new_batch.csv
"""

import argparse
import logging

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)


# ─────────────────────────────────────────────
# Population Stability Index (PSI)
# ─────────────────────────────────────────────


def compute_psi(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    PSI measures how much a feature's distribution has shifted.

    Interpretation (industry standard):
      PSI < 0.10  — No significant change
      PSI < 0.20  — Minor shift, monitor
      PSI >= 0.20 — Major shift → consider retraining

    Args:
        reference: Array from training data (reference distribution)
        current:   Array from new/production data
        n_bins:    Number of bins for binning continuous features

    Returns:
        PSI score (float)
    """
    # Use reference quantiles for consistent binning
    breakpoints = np.nanpercentile(reference, np.linspace(0, 100, n_bins + 1))
    breakpoints = np.unique(breakpoints)  # deduplicate for constant-ish features
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    def bin_counts(arr):
        counts, _ = np.histogram(arr, bins=breakpoints)
        counts = counts + 1e-6  # avoid division by zero
        return counts / counts.sum()

    ref_pct = bin_counts(reference)
    cur_pct = bin_counts(current)

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


def compute_all_psi(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    numeric_features: list[str],
    psi_threshold: float = 0.20,
) -> pd.DataFrame:
    """
    Compute PSI for all numeric features between reference and current datasets.
    Returns a DataFrame sorted by PSI (most drifted first).
    """
    results = []
    common = [
        f
        for f in numeric_features
        if f in reference_df.columns and f in current_df.columns
    ]

    for feat in common:
        psi = compute_psi(
            reference_df[feat].dropna().values,
            current_df[feat].dropna().values,
        )
        results.append(
            {
                "feature": feat,
                "psi": round(psi, 4),
                "status": (
                    "🔴 ALERT"
                    if psi >= psi_threshold
                    else "🟡 Monitor" if psi >= 0.10 else "🟢 Stable"
                ),
            }
        )

    return pd.DataFrame(results).sort_values("psi", ascending=False)


# ─────────────────────────────────────────────
# Prediction drift
# ─────────────────────────────────────────────


def monitor_prediction_drift(
    reference_probs: np.ndarray,
    current_probs: np.ndarray,
    alert_threshold: float = 0.05,
) -> dict:
    """
    Check if the distribution of churn probabilities has shifted.
    A large shift in mean predicted probability often signals concept drift.
    """
    ref_mean = reference_probs.mean()
    cur_mean = current_probs.mean()
    delta = abs(cur_mean - ref_mean)

    return {
        "reference_mean_prob": round(float(ref_mean), 4),
        "current_mean_prob": round(float(cur_mean), 4),
        "delta": round(float(delta), 4),
        "status": "🔴 ALERT" if delta > alert_threshold else "🟢 Stable",
        "recommendation": (
            "Significant shift in predicted churn rate. Consider retraining."
            if delta > alert_threshold
            else "Prediction distribution stable."
        ),
    }


# ─────────────────────────────────────────────
# Full monitoring report
# ─────────────────────────────────────────────


def run_monitoring_report(
    reference_path: str,
    current_path: str,
    config_path: str = "configs/config.yaml",
) -> dict:
    """
    Full drift report comparing reference (training) vs current (new) data.
    Prints a human-readable report and returns structured results.
    """
    import yaml

    from src.features.engineering import ChurnFeatureEngineer

    config = yaml.safe_load(open(config_path))
    reference_df = pd.read_csv(reference_path)
    current_df = pd.read_csv(current_path)

    model = joblib.load(config["api"]["model_path"])
    preprocessor = joblib.load(config["api"]["preprocessor_path"])

    # Feature drift
    numeric_features = config["features"]["numeric_cols"] + [
        "avg_monthly_spend",
        "service_count",
        "is_high_value",
        "contract_risk_score",
        "has_online_services",
    ]

    # Engineer features for both sets
    eng = ChurnFeatureEngineer()
    ref_eng = eng.fit_transform(reference_df.drop(columns=["Churn"], errors="ignore"))
    cur_eng = eng.fit_transform(current_df.drop(columns=["Churn"], errors="ignore"))

    psi_df = compute_all_psi(ref_eng, cur_eng, numeric_features)

    # Prediction drift
    ref_X = preprocessor.transform(ref_eng)
    cur_X = preprocessor.transform(cur_eng)
    ref_probs = model.predict_proba(ref_X)[:, 1]
    cur_probs = model.predict_proba(cur_X)[:, 1]
    pred_drift = monitor_prediction_drift(ref_probs, cur_probs)

    # Print report
    logger.info("\n" + "=" * 55)
    logger.info("MODEL MONITORING REPORT")
    logger.info("=" * 55)
    logger.info(f"Reference: {reference_path} ({len(reference_df):,} rows)")
    logger.info(f"Current:   {current_path} ({len(current_df):,} rows)")
    logger.info("\n--- Feature Drift (PSI) ---")
    logger.info(psi_df.to_string(index=False))
    logger.info("\n--- Prediction Distribution Drift ---")
    for k, v in pred_drift.items():
        logger.info(f"  {k}: {v}")

    alerts = psi_df[psi_df["status"].str.startswith("🔴")]
    if not alerts.empty or pred_drift["status"].startswith("🔴"):
        logger.warning("\n⚠️  RETRAINING RECOMMENDED")
    else:
        logger.info("\n✅ Model appears stable — no retraining needed")

    return {
        "feature_drift": psi_df.to_dict(orient="records"),
        "prediction_drift": pred_drift,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model drift monitoring report")
    parser.add_argument(
        "--reference", required=True, help="Path to reference (training) CSV"
    )
    parser.add_argument("--current", required=True, help="Path to current (new) CSV")
    parser.add_argument("--config", default="configs/config.yaml", help="Config path")
    args = parser.parse_args()

    run_monitoring_report(args.reference, args.current, args.config)
