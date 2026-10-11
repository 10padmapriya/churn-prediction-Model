"""
tests/test_api.py

Integration tests for all FastAPI endpoints.
Uses httpx.AsyncClient with ASGITransport — no running server needed.

Run with: pytest tests/test_api.py -v
"""

import io
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient

# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def client():
    """Synchronous test client — loads model once per module."""
    from src.api.main import app

    with TestClient(app) as c:
        yield c


VALID_CUSTOMER = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 3,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 89.5,
    "TotalCharges": 268.5,
}

LOW_RISK_CUSTOMER = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "Yes",
    "tenure": 60,
    "PhoneService": "Yes",
    "MultipleLines": "Yes",
    "InternetService": "DSL",
    "OnlineSecurity": "Yes",
    "OnlineBackup": "Yes",
    "DeviceProtection": "Yes",
    "TechSupport": "Yes",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Two year",
    "PaperlessBilling": "No",
    "PaymentMethod": "Bank transfer (automatic)",
    "MonthlyCharges": 65.0,
    "TotalCharges": 3900.0,
}


# ── Health endpoint ───────────────────────────────────────────────


class TestHealth:
    def test_health_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_model_loaded(self, client):
        d = client.get("/health").json()
        assert d["status"] == "ok"
        assert d["model_loaded"] is True

    def test_health_has_threshold(self, client):
        d = client.get("/health").json()
        assert "threshold" in d
        assert 0 < d["threshold"] < 1

    def test_health_feature_count(self, client):
        d = client.get("/health").json()
        assert d["feature_count"] == 38


# ── Predict endpoint ──────────────────────────────────────────────


class TestPredict:
    def test_predict_returns_200(self, client):
        r = client.post("/predict", json=VALID_CUSTOMER)
        assert r.status_code == 200

    def test_predict_schema(self, client):
        d = client.post("/predict", json=VALID_CUSTOMER).json()
        assert "churn_probability" in d
        assert "churn_prediction" in d
        assert "risk_tier" in d
        assert "threshold_used" in d
        assert "estimated_clv_at_risk" in d

    def test_predict_probability_in_range(self, client):
        d = client.post("/predict", json=VALID_CUSTOMER).json()
        assert 0.0 <= d["churn_probability"] <= 1.0

    def test_predict_risk_tier_valid(self, client):
        d = client.post("/predict", json=VALID_CUSTOMER).json()
        assert d["risk_tier"] in ("Low", "Medium", "High")

    def test_high_risk_profile(self, client):
        """Month-to-month + Fiber + new customer should be high churn risk."""
        d = client.post("/predict", json=VALID_CUSTOMER).json()
        # This profile is deliberately high-risk — should exceed threshold
        assert d["churn_prediction"] is True
        assert d["churn_probability"] > 0.5

    def test_low_risk_profile(self, client):
        """Two-year contract + long tenure + many services → low risk."""
        d = client.post("/predict", json=LOW_RISK_CUSTOMER).json()
        assert d["churn_prediction"] is False
        assert d["churn_probability"] < 0.5

    def test_predict_clv_at_risk_positive(self, client):
        d = client.post("/predict", json=VALID_CUSTOMER).json()
        assert d["estimated_clv_at_risk"] >= 0

    def test_predict_missing_field_422(self, client):
        bad = {k: v for k, v in VALID_CUSTOMER.items() if k != "tenure"}
        r = client.post("/predict", json=bad)
        assert r.status_code == 422

    def test_predict_invalid_senior_citizen_422(self, client):
        bad = {**VALID_CUSTOMER, "SeniorCitizen": 5}
        r = client.post("/predict", json=bad)
        assert r.status_code == 422

    def test_predict_negative_charges_422(self, client):
        bad = {**VALID_CUSTOMER, "MonthlyCharges": -10}
        r = client.post("/predict", json=bad)
        assert r.status_code == 422


# ── Explain endpoint ──────────────────────────────────────────────


class TestExplain:
    def test_explain_returns_200(self, client):
        r = client.post("/explain", json=VALID_CUSTOMER)
        assert r.status_code == 200

    def test_explain_schema(self, client):
        d = client.post("/explain", json=VALID_CUSTOMER).json()
        assert "churn_probability" in d
        assert "base_value" in d
        assert "top_drivers" in d
        assert isinstance(d["top_drivers"], list)

    def test_explain_drivers_have_required_keys(self, client):
        d = client.post("/explain", json=VALID_CUSTOMER).json()
        for driver in d["top_drivers"]:
            assert "feature" in driver
            assert "raw_shap" in driver
            assert "direction" in driver
            assert driver["direction"] in ("increases_churn", "decreases_churn")

    def test_explain_top_drivers_count(self, client):
        d = client.post("/explain", json=VALID_CUSTOMER).json()
        assert 1 <= len(d["top_drivers"]) <= 10

    def test_explain_probability_matches_predict(self, client):
        prob_predict = client.post("/predict", json=VALID_CUSTOMER).json()[
            "churn_probability"
        ]
        prob_explain = client.post("/explain", json=VALID_CUSTOMER).json()[
            "churn_probability"
        ]
        assert abs(prob_predict - prob_explain) < 1e-4


# ── Model performance endpoint ────────────────────────────────────


class TestModelPerformance:
    def test_performance_returns_200(self, client):
        r = client.get("/model/performance")
        assert r.status_code == 200

    def test_performance_has_best_model(self, client):
        d = client.get("/model/performance").json()
        assert "best_model" in d
        assert d["best_model"] in ("LogisticRegression", "XGBoost", "LightGBM")

    def test_performance_has_model_metrics(self, client):
        d = client.get("/model/performance").json()
        assert "models" in d
        for m in d["models"].values():
            assert "roc_auc" in m
            assert "recall" in m
            assert "net_value_usd" in m
            assert 0.5 < m["roc_auc"] < 1.0

    def test_performance_best_model_meets_quality_gate(self, client):
        d = client.get("/model/performance").json()
        best = d["models"][d["best_model"]]
        assert best["roc_auc"] >= 0.80, f"AUC {best['roc_auc']:.3f} below 0.80"
        assert best["recall"] >= 0.75, f"Recall {best['recall']:.3f} below 0.75"


# ── Batch prediction endpoint ─────────────────────────────────────


class TestBatch:
    def _make_csv(self, n=10):
        """Build a minimal valid CSV for batch prediction."""
        import pandas as pd

        rows = [VALID_CUSTOMER.copy() for _ in range(n // 2)] + [
            LOW_RISK_CUSTOMER.copy() for _ in range(n - n // 2)
        ]
        buf = io.StringIO()
        pd.DataFrame(rows).to_csv(buf, index=False)
        return buf.getvalue().encode()

    def test_batch_returns_200(self, client):
        csv_bytes = self._make_csv(10)
        r = client.post(
            "/predict/batch",
            files={"file": ("test.csv", csv_bytes, "text/csv")},
        )
        assert r.status_code == 200

    def test_batch_schema(self, client):
        csv_bytes = self._make_csv(10)
        d = client.post(
            "/predict/batch",
            files={"file": ("test.csv", csv_bytes, "text/csv")},
        ).json()
        assert d["total_customers"] == 10
        assert "predicted_churners" in d
        assert "churn_rate" in d
        assert "total_revenue_at_risk" in d
        assert len(d["predictions"]) == 10

    def test_batch_predictions_have_fields(self, client):
        csv_bytes = self._make_csv(5)
        d = client.post(
            "/predict/batch",
            files={"file": ("test.csv", csv_bytes, "text/csv")},
        ).json()
        for pred in d["predictions"]:
            assert "churn_probability" in pred
            assert "churn_prediction" in pred
            assert "risk_tier" in pred
            assert "clv_at_risk" in pred

    def test_batch_high_risk_majority(self, client):
        """All VALID_CUSTOMER rows should be flagged as churn."""
        csv_bytes = self._make_csv(10)
        d = client.post(
            "/predict/batch",
            files={"file": ("test.csv", csv_bytes, "text/csv")},
        ).json()
        # At least half should be flagged (we have 5 high-risk rows)
        assert d["predicted_churners"] >= 4
