"""
src/data/ingestion.py

Handles raw data loading, cleaning, and validation.
All transformations here are deterministic and reversible —
no feature engineering or encoding happens at this stage.
"""

import pandas as pd
import numpy as np
import yaml
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_raw_data(path: str) -> pd.DataFrame:
    """Load raw CSV and perform basic validation."""
    logger.info(f"Loading raw data from {path}")
    df = pd.read_csv(path)
    logger.info(f"Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix known issues in the Telco dataset:
    - TotalCharges is object dtype due to whitespace strings
    - Impute the 11 rows where TotalCharges is blank (new customers, tenure=0)
    """
    df = df.copy()

    # TotalCharges: coerce to float, whitespace → NaN
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # For tenure=0, TotalCharges should logically equal MonthlyCharges
    mask = df["TotalCharges"].isna()
    logger.info(f"Imputing {mask.sum()} missing TotalCharges (tenure=0 customers)")
    df.loc[mask, "TotalCharges"] = df.loc[mask, "MonthlyCharges"]

    # Standardize binary target to 0/1 int
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    # Drop customerID — identifier, not a feature
    df = df.drop(columns=["customerID"])

    logger.info(f"Churn rate: {df['Churn'].mean():.1%}")
    return df


def validate_data(df: pd.DataFrame) -> None:
    """Assert data integrity after cleaning."""
    assert df.isnull().sum().sum() == 0, "Nulls remain after cleaning"
    assert df["Churn"].isin([0, 1]).all(), "Target is not binary"
    assert (df["TotalCharges"] >= 0).all(), "Negative TotalCharges found"
    assert (df["MonthlyCharges"] > 0).all(), "Non-positive MonthlyCharges found"
    logger.info("Data validation passed ✓")


def run_ingestion(config_path: str = "configs/config.yaml") -> pd.DataFrame:
    config = load_config(config_path)
    df = load_raw_data(config["data"]["raw_path"])
    df = clean_data(df)
    validate_data(df)

    out_path = Path(config["data"]["processed_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info(f"Cleaned data saved → {out_path}")
    return df


if __name__ == "__main__":
    run_ingestion()
