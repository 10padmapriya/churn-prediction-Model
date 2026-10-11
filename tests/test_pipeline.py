"""
tests/test_pipeline.py

Tests for ingestion, feature engineering, and API logic.
Run with: pytest tests/ -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.ingestion import clean_data, load_config, validate_data
from src.features.engineering import ChurnFeatureEngineer, prepare_features
from src.models.train import optimize_threshold

# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def raw_df():
    return pd.read_csv("data/raw/telco_churn.csv")


@pytest.fixture
def clean_df(raw_df):
    return clean_data(raw_df)


@pytest.fixture
def config():
    return load_config("configs/config.yaml")


# ── Data ingestion tests ──────────────────────────────────────────


class TestIngestion:
    def test_shape(self, raw_df):
        assert raw_df.shape == (7043, 21)

    def test_clean_no_nulls(self, clean_df):
        assert clean_df.isnull().sum().sum() == 0

    def test_target_is_binary(self, clean_df):
        assert set(clean_df["Churn"].unique()) == {0, 1}

    def test_churn_rate(self, clean_df):
        rate = clean_df["Churn"].mean()
        assert 0.25 < rate < 0.28, f"Churn rate {rate:.3f} out of expected range"

    def test_total_charges_imputed(self, clean_df):
        assert clean_df["TotalCharges"].isnull().sum() == 0
        assert (clean_df["TotalCharges"] >= 0).all()

    def test_validate_passes(self, clean_df):
        # Should not raise
        validate_data(clean_df)


# ── Feature engineering tests ─────────────────────────────────────


class TestFeatureEngineering:
    def test_engineer_adds_columns(self, clean_df):
        df = clean_df.drop(columns=["Churn"])
        eng = ChurnFeatureEngineer()
        out = eng.fit_transform(df)
        for col in [
            "tenure_group",
            "service_count",
            "is_high_value",
            "contract_risk_score",
            "avg_monthly_spend",
            "has_online_services",
        ]:
            assert col in out.columns, f"Missing engineered column: {col}"

    def test_service_count_range(self, clean_df):
        df = clean_df.drop(columns=["Churn"])
        eng = ChurnFeatureEngineer()
        out = eng.fit_transform(df)
        assert out["service_count"].between(0, 6).all()

    def test_contract_risk_score_values(self, clean_df):
        df = clean_df.drop(columns=["Churn"])
        eng = ChurnFeatureEngineer()
        out = eng.fit_transform(df)
        assert set(out["contract_risk_score"].unique()).issubset({0, 1, 2})

    def test_prepare_features_shape(self, clean_df, config):
        X, y, _prep, names, _eng = prepare_features(clean_df, config, fit=True)
        assert X.shape[0] == len(clean_df)
        assert X.shape[1] == len(names)
        assert len(y) == len(clean_df)

    def test_no_nulls_in_processed(self, clean_df, config):
        X, _y, _, _, _ = prepare_features(clean_df, config, fit=True)
        assert not np.isnan(X).any()


# ── Threshold optimization tests ──────────────────────────────────


class TestThresholdOptimization:
    def test_threshold_in_valid_range(self):
        np.random.seed(42)
        y_true = np.random.binomial(1, 0.3, 500)
        y_prob = np.clip(y_true + np.random.normal(0, 0.3, 500), 0, 1)
        t, cost = optimize_threshold(y_true, y_prob, cost_fn=500, cost_fp=20)
        assert 0.05 <= t <= 0.95
        assert cost >= 0

    def test_high_fn_cost_lowers_threshold(self):
        """High FN cost should push threshold down (more aggressive recall)."""
        np.random.seed(42)
        y_true = np.random.binomial(1, 0.3, 1000)
        y_prob = np.clip(y_true * 0.7 + np.random.uniform(0, 0.3, 1000), 0, 1)
        t_high_fn, _ = optimize_threshold(y_true, y_prob, cost_fn=1000, cost_fp=20)
        t_low_fn, _ = optimize_threshold(y_true, y_prob, cost_fn=50, cost_fp=20)
        assert t_high_fn <= t_low_fn


# ── Config tests ──────────────────────────────────────────────────


class TestConfig:
    def test_config_loads(self, config):
        assert "data" in config
        assert "models" in config
        assert "business" in config
        assert "features" in config

    def test_cost_matrix_sensible(self, config):
        assert (
            config["business"]["cost_false_negative"]
            > config["business"]["cost_false_positive"]
        )

    def test_data_paths_defined(self, config):
        assert "raw_path" in config["data"]
        assert "processed_path" in config["data"]
