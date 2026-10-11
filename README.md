# Churn Prediction + Business Impact Dashboard

![CI](https://github.com/your-username/churn-predictor/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.12-blue)
![Tests](https://img.shields.io/badge/tests-58%20passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue)

An end-to-end ML system that predicts customer churn using **cost-sensitive classification**, served by a **FastAPI backend** with a **React dashboard** showing live business impact in dollar terms.

> **Interview answer:** "I framed churn as a cost-sensitive problem — a missed churner costs $500 in lost CLV while a false alarm costs $20 in retention spend. I optimized the decision threshold for that asymmetric cost matrix, shifting recall from 80% to 99% with $555K net value on the test set."

---

## Results

| Model | CV AUC | Test AUC | Recall | Net Value |
|---|---|---|---|---|
| **Logistic Regression** ⭐ | 0.852 | **0.846** | **98.7%** | **$555K** |
| XGBoost | 0.880 | 0.836 | 96.5% | $543K |
| LightGBM | 0.880 | 0.831 | 94.1% | $530K |

> XGBoost has higher CV AUC but a 0.044 gap to holdout — Logistic Regression generalises best (gap: 0.006) and is better calibrated for CLV estimation.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│            React Dashboard (Vite + Recharts)        │
│  Business Impact · Model Perf · Live Predict · Batch│
└────────────────────┬────────────────────────────────┘
                     │ HTTP/JSON
┌────────────────────▼────────────────────────────────┐
│              FastAPI Backend (Python 3.12)           │
│  /predict  /explain  /predict/batch  /health        │
│  /model/performance                                  │
└────────────────────┬────────────────────────────────┘
                     │ joblib artifacts
┌────────────────────▼────────────────────────────────┐
│              ML Pipeline                             │
│  Ingestion → Feature Eng. → SMOTE → Models → SHAP  │
│  Cost-matrix threshold optimisation → Monitor (PSI) │
└─────────────────────────────────────────────────────┘
```

---

## Project Structure

```
churn-predictor/
├── .github/workflows/ci.yml       # 6-stage CI: lint → test → train → API → Docker → Frontend
├── configs/config.yaml            # All hyperparameters, cost matrix, paths
├── data/
│   ├── raw/                       # Original Telco CSV (gitignored)
│   └── processed/                 # Model artifacts + SHAP charts (gitignored)
├── notebooks/
│   ├── 01_eda.ipynb               # Business-framed EDA
│   ├── 02_shap_analysis.ipynb     # Global + per-customer SHAP
│   └── 03_model_selection.ipynb   # CV vs holdout, calibration, scorecard
├── src/
│   ├── data/ingestion.py          # Load, clean, validate
│   ├── features/engineering.py    # ChurnFeatureEngineer + sklearn pipeline
│   ├── models/train.py            # Multi-model + cost-sensitive threshold
│   ├── models/explain.py          # SHAP per-customer explanations
│   ├── models/monitor.py          # PSI drift detection
│   └── api/main.py                # FastAPI (5 endpoints)
├── frontend/
│   ├── src/
│   │   ├── api.js                 # Centralised fetch client
│   │   ├── hooks.js               # React hooks for each endpoint
│   │   └── ChurnDashboard.jsx     # 4-tab dashboard (fully wired)
│   ├── Dockerfile                 # nginx production build
│   └── package.json
├── tests/
│   ├── test_pipeline.py           # 16 tests: ingestion, features, threshold, config
│   ├── test_api.py                # 27 tests: all 5 API endpoints
│   └── test_monitor.py            # 15 tests: PSI + prediction drift
├── train_pipeline.py              # Master script: runs full pipeline end-to-end
├── MODEL_CARD.md                  # Mitchell et al. (2019) framework
├── Dockerfile                     # API container
└── docker-compose.yml             # API + frontend services
```

---

## Quickstart

### 1. Install & train

```bash
git clone https://github.com/your-username/churn-predictor
cd churn-predictor

pip install -r requirements.txt

# Downloads dataset, trains 3 models, writes optimal threshold to config
python train_pipeline.py
```

### 2. Run the API

```bash
uvicorn src.api.main:app --reload --port 8000
# → http://localhost:8000/docs  (interactive Swagger UI)
```

### 3. Run the dashboard

```bash
cd frontend
cp .env.example .env.local        # VITE_API_URL=http://localhost:8000
npm install
npm run dev
# → http://localhost:5173
```

### 4. Docker (full stack)

```bash
docker-compose up --build
# API  → http://localhost:8000
# UI   → http://localhost:5173
```

### 5. Run tests

```bash
pytest tests/ -v                  # 58 tests
```

---

## API Reference

### `POST /predict`

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"tenure": 3, "Contract": "Month-to-month", "MonthlyCharges": 89.5, ...}'
```

```json
{
  "churn_probability": 0.8821,
  "churn_prediction": true,
  "risk_tier": "High",
  "threshold_used": 0.1304,
  "estimated_clv_at_risk": 1376.08
}
```

### `POST /explain`
Returns SHAP-based top-10 feature drivers for one customer. Powers the waterfall chart in the dashboard.

### `POST /predict/batch`
Accepts CSV upload. Returns aggregate stats + per-row risk scores.

### `GET /model/performance`
Returns stored evaluation metrics including confusion matrix, business impact figures, and all three model comparisons.

### `GET /health`
Returns model load status, feature count, and current threshold.

---

## Key Engineering Decisions

### Cost-sensitive threshold optimisation
Default classifiers use threshold=0.5 (optimises accuracy). This project minimises:

```
Total cost = (FN × $500) + (FP × $20)
```

Optimal threshold: **0.13**, found by grid search over [0.05, 0.90]. This shifts recall from 80% → 99% (vs. the default 0.5 threshold).

### No-leakage feature engineering
`ChurnFeatureEngineer` is a proper sklearn transformer. It's fit exclusively on training data and **persisted to `data/processed/engineer.joblib`**. The API loads this artifact — it does not refit on single-customer inputs (which would make `is_high_value` always 0).

### SMOTE: train only, never test
Applied with `sampling_strategy=0.5` (not full resampling to avoid overfit). Test set is never touched.

### Threshold written to config after training
`train_pipeline.py` writes the optimal threshold back to `configs/config.yaml`. The API reads from there. Single source of truth.

---

## Monitoring

```bash
python src/models/monitor.py \
  --reference data/processed/telco_processed.csv \
  --current data/new_batch.csv
```

Reports PSI (Population Stability Index) per feature and prediction distribution drift. PSI > 0.20 triggers a retraining recommendation.

---

## Skills Demonstrated

| Skill | Where |
|---|---|
| Cost-sensitive ML | `src/models/train.py` — `optimize_threshold()` |
| sklearn custom transformer | `src/features/engineering.py` — `ChurnFeatureEngineer` |
| SHAP explainability | `src/models/explain.py` + `/explain` endpoint |
| FastAPI + Pydantic v2 | `src/api/main.py` — 5 endpoints, lifespan pattern |
| React + custom hooks | `frontend/src/hooks.js` — no mock data |
| PSI drift monitoring | `src/models/monitor.py` |
| CI/CD with quality gates | `.github/workflows/ci.yml` — 6 jobs |
| Docker multi-stage build | `frontend/Dockerfile` — node build → nginx |
| Model card | `MODEL_CARD.md` — Mitchell et al. framework |

---

*Dataset: IBM Telco Customer Churn (public domain)*  
*Author: Padmapriya · NYU Center for Data Science · 2026*
