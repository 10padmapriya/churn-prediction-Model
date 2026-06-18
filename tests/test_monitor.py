"""
tests/test_monitor.py

Tests for PSI-based drift detection and prediction monitoring.
Run with: pytest tests/test_monitor.py -v
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.monitor import (
    compute_psi,
    compute_all_psi,
    monitor_prediction_drift,
)


class TestPSI:
    """Population Stability Index correctness tests."""

    def test_identical_distributions_near_zero(self):
        """Same distribution → PSI should be ~0."""
        np.random.seed(42)
        ref = np.random.normal(65, 20, 5000)
        cur = np.random.normal(65, 20, 5000)
        psi = compute_psi(ref, cur)
        assert psi < 0.05, f"Identical distributions PSI={psi:.4f}, expected < 0.05"

    def test_shifted_distribution_high_psi(self):
        """Mean shift of 2σ → PSI should exceed alert threshold (0.20)."""
        np.random.seed(42)
        ref = np.random.normal(65, 20, 5000)
        cur = np.random.normal(105, 20, 1000)  # +2σ shift
        psi = compute_psi(ref, cur)
        assert psi > 0.20, f"Shifted distribution PSI={psi:.4f}, expected > 0.20"

    def test_psi_non_negative(self):
        """PSI is always ≥ 0 by definition."""
        np.random.seed(0)
        for _ in range(5):
            ref = np.random.exponential(50, 1000)
            cur = np.random.exponential(60, 500)
            assert compute_psi(ref, cur) >= 0

    def test_psi_symmetric_direction(self):
        """Slight shift should still give low-ish PSI (< 0.10)."""
        np.random.seed(42)
        ref = np.random.normal(65, 20, 5000)
        cur = np.random.normal(68, 20, 1000)  # very small shift
        psi = compute_psi(ref, cur)
        assert psi < 0.10, f"Small shift PSI={psi:.4f}, expected < 0.10"

    def test_psi_custom_bins(self):
        """n_bins parameter should be respected without error."""
        ref = np.random.normal(0, 1, 1000)
        cur = np.random.normal(0, 1, 1000)
        for n in [5, 10, 20]:
            psi = compute_psi(ref, cur, n_bins=n)
            assert psi >= 0

    def test_compute_all_psi_returns_dataframe(self):
        """compute_all_psi should return a sorted DataFrame."""
        np.random.seed(42)
        ref = pd.DataFrame({
            "MonthlyCharges": np.random.normal(65, 20, 500),
            "tenure":         np.random.exponential(30, 500),
            "TotalCharges":   np.random.normal(1500, 800, 500),
        })
        cur = pd.DataFrame({
            "MonthlyCharges": np.random.normal(75, 25, 200),  # drifted
            "tenure":         np.random.exponential(30, 200),
            "TotalCharges":   np.random.normal(1500, 800, 200),
        })
        result = compute_all_psi(ref, cur, ["MonthlyCharges", "tenure", "TotalCharges"])
        assert isinstance(result, pd.DataFrame)
        assert "feature" in result.columns
        assert "psi" in result.columns
        assert "status" in result.columns
        assert len(result) == 3
        # Should be sorted descending by PSI
        assert result["psi"].is_monotonic_decreasing

    def test_compute_all_psi_alert_status(self):
        """A heavily drifted feature should get ALERT status."""
        np.random.seed(42)
        ref = pd.DataFrame({"feat": np.random.normal(0, 1, 1000)})
        cur = pd.DataFrame({"feat": np.random.normal(5, 1, 200)})  # big shift
        result = compute_all_psi(ref, cur, ["feat"], psi_threshold=0.20)
        assert result.iloc[0]["status"].startswith("🔴")

    def test_compute_all_psi_stable_status(self):
        """A stable feature should get Stable status."""
        np.random.seed(42)
        ref = pd.DataFrame({"feat": np.random.normal(0, 1, 1000)})
        cur = pd.DataFrame({"feat": np.random.normal(0, 1, 500)})
        result = compute_all_psi(ref, cur, ["feat"])
        assert result.iloc[0]["status"].startswith("🟢")

    def test_ignores_missing_columns(self):
        """Features not in both DataFrames should be silently skipped."""
        ref = pd.DataFrame({"a": np.random.normal(0, 1, 100), "b": np.random.normal(0, 1, 100)})
        cur = pd.DataFrame({"a": np.random.normal(0, 1, 50)})  # b missing
        result = compute_all_psi(ref, cur, ["a", "b", "c_nonexistent"])
        assert len(result) == 1
        assert result.iloc[0]["feature"] == "a"


class TestPredictionDrift:
    """Prediction distribution monitoring tests."""

    def test_stable_distributions_no_alert(self):
        np.random.seed(42)
        ref = np.random.uniform(0.1, 0.4, 1000)
        cur = np.random.uniform(0.1, 0.4, 500)
        result = monitor_prediction_drift(ref, cur)
        assert result["status"].startswith("🟢")

    def test_drifted_distributions_alert(self):
        np.random.seed(42)
        ref = np.random.uniform(0.1, 0.3, 1000)
        cur = np.random.uniform(0.5, 0.8, 500)
        result = monitor_prediction_drift(ref, cur)
        assert result["status"].startswith("🔴")

    def test_result_has_required_keys(self):
        ref = np.array([0.2, 0.3, 0.25])
        cur = np.array([0.5, 0.6, 0.55])
        result = monitor_prediction_drift(ref, cur)
        for key in ["reference_mean_prob", "current_mean_prob", "delta", "status", "recommendation"]:
            assert key in result

    def test_delta_is_absolute(self):
        """Delta should always be positive."""
        ref = np.array([0.6, 0.7])
        cur = np.array([0.2, 0.3])
        result = monitor_prediction_drift(ref, cur)
        assert result["delta"] >= 0

    def test_custom_alert_threshold(self):
        """Very tight threshold should trigger alert on small drift."""
        ref = np.array([0.30, 0.31, 0.29])
        cur = np.array([0.35, 0.36, 0.34])
        result = monitor_prediction_drift(ref, cur, alert_threshold=0.01)
        assert result["status"].startswith("🔴")

    def test_mean_probs_rounded(self):
        ref = np.array([0.25])
        cur = np.array([0.75])
        result = monitor_prediction_drift(ref, cur)
        assert result["reference_mean_prob"] == 0.25
        assert result["current_mean_prob"] == 0.75
