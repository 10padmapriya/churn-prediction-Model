"""
src/api/main.py

FastAPI backend for the churn prediction service.
Endpoints:
  GET  /health              — service health + model metadata
  POST /predict             — single customer churn probability
  POST /predict/batch       — batch predictions (CSV upload)
  POST /explain             — SHAP explanation for one customer
  GET  /model/performance   — stored eval metrics for the dashboard
"""

import io
import json
import logging

# Local imports
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import joblib
import numpy as np
import pandas as pd
import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.features.engineering import ChurnFeatureEngineer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ─────────────────────────────────────────────
# App state (loaded at startup)
# ─────────────────────────────────────────────

app_state = {}


def load_artifacts():
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)
    model = joblib.load(config["api"]["model_path"])
    preprocessor = joblib.load(config["api"]["preprocessor_path"])
    feature_names = joblib.load(config["api"]["feature_names_path"])

    # Load stored metrics if available
    metrics_path = Path("data/processed/eval_metrics.json")
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}

    from pathlib import Path as _Path

    engineer_path = _Path("data/processed/engineer.joblib")
    engineer = joblib.load(engineer_path) if engineer_path.exists() else None

    return {
        "config": config,
        "model": model,
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "engineer": engineer,
        "threshold": config["training"]["default_threshold"],
        "metrics": metrics,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading model artifacts...")
    app_state.update(load_artifacts())
    logger.info(f"Model loaded. Threshold={app_state['threshold']:.3f}")
    yield
    app_state.clear()


# ─────────────────────────────────────────────
# App init
# ─────────────────────────────────────────────

app = FastAPI(
    title="Churn Prediction API",
    description="Predict customer churn probability with SHAP-based explanations.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# Pydantic schemas
# ─────────────────────────────────────────────


class CustomerFeatures(BaseModel):
    gender: str = Field(...)
    SeniorCitizen: int = Field(..., ge=0, le=1)
    Partner: str = Field(...)
    Dependents: str = Field(...)
    tenure: int = Field(..., ge=0)
    PhoneService: str = Field(...)
    MultipleLines: str = Field(...)
    InternetService: str = Field(...)
    OnlineSecurity: str = Field(...)
    OnlineBackup: str = Field(...)
    DeviceProtection: str = Field(...)
    TechSupport: str = Field(...)
    StreamingTV: str = Field(...)
    StreamingMovies: str = Field(...)
    Contract: str = Field(...)
    PaperlessBilling: str = Field(...)
    PaymentMethod: str = Field(...)
    MonthlyCharges: float = Field(..., gt=0)
    TotalCharges: float = Field(..., ge=0)


class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: bool
    risk_tier: str  # "Low" / "Medium" / "High"
    threshold_used: float
    estimated_clv_at_risk: float


class ExplainResponse(BaseModel):
    churn_probability: float
    base_value: float
    top_drivers: list[dict]


# ─────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────


def customer_to_df(customer: CustomerFeatures) -> pd.DataFrame:
    return pd.DataFrame([customer.model_dump()])


def preprocess_input(df: pd.DataFrame) -> np.ndarray:
    engineer = app_state.get("engineer")
    if engineer is None:
        # Fallback: fit on this input (only safe if engineer.joblib not available)
        engineer = ChurnFeatureEngineer()
        df_eng = engineer.fit_transform(df)
    else:
        df_eng = engineer.transform(df)
    return app_state["preprocessor"].transform(df_eng)


def risk_tier(prob: float) -> str:
    if prob < 0.35:
        return "Low"
    elif prob < 0.65:
        return "Medium"
    return "High"


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": "model" in app_state,
        "threshold": app_state.get("threshold"),
        "feature_count": len(app_state.get("feature_names", [])),
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(customer: CustomerFeatures):
    try:
        df = customer_to_df(customer)
        X = preprocess_input(df)
        prob = float(app_state["model"].predict_proba(X)[0, 1])
        threshold = app_state["threshold"]
        config = app_state["config"]

        avg_clv = (
            config["business"]["avg_monthly_revenue"]
            * config["business"]["avg_clv_months"]
        )
        clv_at_risk = prob * avg_clv

        return PredictionResponse(
            churn_probability=round(prob, 4),
            churn_prediction=prob >= threshold,
            risk_tier=risk_tier(prob),
            threshold_used=threshold,
            estimated_clv_at_risk=round(clv_at_risk, 2),
        )
    except Exception as e:
        logger.exception("Prediction error")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/predict/batch")
async def predict_batch(file: Annotated[UploadFile, File()]):
    """Accept CSV upload, return predictions for all rows."""
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))

        # Drop target and ID cols if present
        df = df.drop(columns=["Churn", "customerID"], errors="ignore")

        engineer = app_state.get("engineer")
        if engineer is None:
            engineer = ChurnFeatureEngineer()
            df_eng = engineer.fit_transform(df)
        else:
            df_eng = engineer.transform(df)
        X = app_state["preprocessor"].transform(df_eng)

        probs = app_state["model"].predict_proba(X)[:, 1]
        threshold = app_state["threshold"]
        config = app_state["config"]
        avg_clv = (
            config["business"]["avg_monthly_revenue"]
            * config["business"]["avg_clv_months"]
        )

        results = pd.DataFrame(
            {
                "churn_probability": probs.round(4),
                "churn_prediction": probs >= threshold,
                "risk_tier": [risk_tier(p) for p in probs],
                "clv_at_risk": (probs * avg_clv).round(2),
            }
        )

        return {
            "total_customers": len(results),
            "predicted_churners": int(results["churn_prediction"].sum()),
            "churn_rate": round(results["churn_prediction"].mean(), 4),
            "total_revenue_at_risk": round(results["clv_at_risk"].sum(), 2),
            "predictions": results.to_dict(orient="records"),
        }
    except Exception as e:
        logger.exception("Batch prediction error")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/explain", response_model=ExplainResponse)
async def explain(customer: CustomerFeatures):
    """Return SHAP-based explanation for one customer."""
    try:
        from src.models.explain import explain_single_customer, get_explainer

        df = customer_to_df(customer)
        X = preprocess_input(df)
        model = app_state["model"]
        feature_names = app_state["feature_names"]

        prob = float(model.predict_proba(X)[0, 1])

        # Build explainer with single sample as background (lightweight)
        explainer = get_explainer(model, X)
        explanation = explain_single_customer(explainer, X[0], feature_names, prob)

        return ExplainResponse(
            churn_probability=round(prob, 4),
            base_value=round(explanation["base_value"], 4),
            top_drivers=explanation["top_drivers"],
        )
    except Exception as e:
        logger.exception("Explain error")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/model/performance")
async def model_performance():
    """Return stored evaluation metrics for the dashboard."""
    metrics = app_state.get("metrics", {})
    if not metrics:
        raise HTTPException(
            status_code=404, detail="Metrics not found. Run training pipeline first."
        )
    return metrics
