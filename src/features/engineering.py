"""
src/features/engineering.py

Feature engineering and sklearn-compatible preprocessing pipeline.
All transformations are fit on train set only — no leakage.
"""

import pandas as pd
import numpy as np
import yaml
import joblib
import logging
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Custom transformer: domain feature engineering
# ─────────────────────────────────────────────

class ChurnFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Adds business-meaningful features before encoding.
    These encode domain knowledge about churn drivers:
    - Tenure stage (new customers churn more)
    - Service bundle (more services = more sticky)
    - Contract risk (month-to-month is highest risk)
    - Value tier (high spenders need special retention)
    """

    def fit(self, X, y=None):
        self.monthly_charges_75th_ = X["MonthlyCharges"].quantile(0.75)
        return self

    def is_fitted(self):
        return hasattr(self, "monthly_charges_75th_")

    def transform(self, X):
        X = X.copy()

        # Tenure group: encodes lifecycle stage
        X["tenure_group"] = pd.cut(
            X["tenure"],
            bins=[0, 12, 24, 48, 72],
            labels=["new", "growing", "loyal", "veteran"],
            include_lowest=True,
        ).astype(str)

        # Avg monthly spend: TotalCharges can hide anomalies
        X["avg_monthly_spend"] = X["TotalCharges"] / (X["tenure"] + 1)

        # Count of add-on services: more services = more locked in
        service_cols = [
            "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies",
        ]
        X["service_count"] = (X[service_cols] == "Yes").sum(axis=1)

        # Has at least one security/backup service
        X["has_online_services"] = (
            (X["OnlineSecurity"] == "Yes") | (X["OnlineBackup"] == "Yes")
        ).astype(int)

        # High-value customer flag
        X["is_high_value"] = (
            X["MonthlyCharges"] > self.monthly_charges_75th_
        ).astype(int)

        # Contract risk score (higher = riskier)
        contract_risk = {
            "Month-to-month": 2,
            "One year": 1,
            "Two year": 0,
        }
        X["contract_risk_score"] = X["Contract"].map(contract_risk).fillna(1)

        return X


# ─────────────────────────────────────────────
# Build full sklearn preprocessing pipeline
# ─────────────────────────────────────────────

def build_preprocessor(config: dict) -> ColumnTransformer:
    """
    Returns a ColumnTransformer that:
    - Standard-scales numeric features
    - One-hot encodes categoricals (drop='first' to avoid multicollinearity)
    """
    numeric_cols = config["features"]["numeric_cols"] + [
        "avg_monthly_spend", "service_count",
        "has_online_services", "is_high_value", "contract_risk_score",
    ]

    categorical_cols = (
        config["features"]["onehot_cols"]
        + config["features"]["binary_cols"][:-1]  # exclude Churn
        + ["tenure_group"]
    )

    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
    )

    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Extract human-readable feature names after fitting."""
    num_features = preprocessor.transformers_[0][2]
    cat_features = (
        preprocessor.named_transformers_["cat"]
        .named_steps["onehot"]
        .get_feature_names_out(preprocessor.transformers_[1][2])
        .tolist()
    )
    return list(num_features) + cat_features


def prepare_features(
    df: pd.DataFrame,
    config: dict,
    fit: bool = True,
    preprocessor=None,
    engineer=None,
) -> tuple:
    """
    Full pipeline: engineer features → fit/transform preprocessor.
    Returns (X_array, y_series, fitted_preprocessor, feature_names, fitted_engineer).

    When fit=True:  a new ChurnFeatureEngineer is fit on df and returned.
    When fit=False: pass in the engineer fitted on training data to avoid leakage.
    """
    target = config["data"]["target_column"]
    y = df[target].copy()
    X = df.drop(columns=[target])

    # Step 1: domain feature engineering
    if fit:
        engineer = ChurnFeatureEngineer()
        X = engineer.fit_transform(X)
    else:
        if engineer is None:
            raise ValueError("Must pass a fitted engineer when fit=False to prevent leakage.")
        X = engineer.transform(X)

    # Step 2: sklearn preprocessing
    if fit:
        preprocessor = build_preprocessor(config)
        X_processed = preprocessor.fit_transform(X)
    else:
        X_processed = preprocessor.transform(X)

    feature_names = get_feature_names(preprocessor)
    logger.info(f"Feature matrix: {X_processed.shape[0]:,} rows × {X_processed.shape[1]} features")

    return X_processed, y, preprocessor, feature_names, engineer
