# Model Card — Telco Customer Churn Predictor

**Version:** 1.0.0  
**Date:** June 2026  
**Author:** Padmapriya — NYU Center for Data Science

---

## Model Summary

A cost-sensitive binary classifier that predicts the probability of customer churn for a telecom company. The primary business goal is to identify at-risk customers *before* they cancel, enabling proactive retention outreach with measurable ROI.

---

## Model Details

| Attribute | Value |
|---|---|
| Model type | Logistic Regression (L2 regularization) |
| Input features | 38 (after engineering from 19 raw features) |
| Output | Churn probability ∈ [0, 1] + binary prediction |
| Decision threshold | 0.13 (cost-optimized; see below) |
| Training data | IBM Telco Customer Churn — 5,634 training samples |
| Test data | 1,409 held-out samples (stratified split) |
| Framework | scikit-learn 1.3.2, Python 3.12 |

---

## Intended Use

**Primary use case:** Internal retention team tooling. Flag customers with high churn probability for targeted outreach before they cancel.

**Appropriate uses:**
- Prioritizing which customers receive retention calls or discount offers
- Monthly/weekly churn risk reports for retention managers
- Input signal for downstream customer lifetime value modeling

**Out-of-scope uses:**
- Real-time pricing decisions
- Credit scoring or credit-adjacent decisions
- Any legally regulated use case (FCRA, ECOA, etc.)
- Customer-facing communications (do not tell a customer they are flagged)

---

## Performance Metrics

| Metric | Value | Context |
|---|---|---|
| ROC-AUC | 0.846 | Averaged over all thresholds |
| Recall | 98.7% | At optimized threshold (0.13) |
| Precision | 36.3% | ~1 in 3 flagged customers actually churns |
| CV AUC (5-fold) | 0.852 ± 0.007 | Stable generalization |
| Churners caught | 369 / 374 | On test set |
| Net business value | $555,320 | Revenue saved minus retention cost |

### Why precision is low by design

The cost matrix used to optimize this model:
- **False Negative** (missed churner): **$500** — lost CLV (24 months × $65 avg monthly charges, discounted)
- **False Positive** (false alarm): **$20** — cost of one proactive retention call/email

A false negative is **25× more costly** than a false positive. Therefore the model is deliberately tuned for high recall (catch almost all churners) at the expense of precision. Retention teams should expect to contact roughly 3 customers for every 1 actual churner.

---

## Training Data

- **Source:** IBM Telco Customer Churn dataset (public, research use)
- **Size:** 7,043 customer records, 21 raw features
- **Churn rate:** 26.5% (1,869 churned / 5,174 retained)
- **Class imbalance handling:** SMOTE applied to training set only (`sampling_strategy=0.5`), bringing minority class to 50% of majority. Test set is untouched — no data leakage.
- **Geography:** US telecom customers (generalization to other markets not validated)

### Known data limitations
- Missing TotalCharges for 11 tenure=0 customers, imputed as MonthlyCharges
- No timestamp data — this is a cross-sectional snapshot, not longitudinal
- No customer support interaction data, which is often predictive of churn

---

## Feature Engineering

Six domain-driven features were engineered based on EDA findings:

| Feature | Logic | Motivation |
|---|---|---|
| `contract_risk_score` | Month-to-month=2, One year=1, Two year=0 | Month-to-month customers churn at 43% vs 11% for two-year |
| `tenure_group` | Bucketed into new/growing/loyal/veteran | First-year churn rate of 48% justifies non-linear treatment |
| `service_count` | Count of 6 add-on services subscribed | 0 services → 36% churn; 6 services → 8% churn |
| `avg_monthly_spend` | TotalCharges / (tenure + 1) | Decorrelates TotalCharges from tenure (Pearson r = 0.83) |
| `is_high_value` | MonthlyCharges > 75th percentile | High-value Fiber optic customers simultaneously at-risk |
| `has_online_services` | OnlineSecurity OR OnlineBackup = Yes | Protective compound feature |

All engineered features are computed inside `ChurnFeatureEngineer`, a proper sklearn transformer that is fit exclusively on training data to prevent leakage.

---

## SHAP Feature Importance (Top 5)

| Feature | Mean |SHAP| | Direction |
|---|---|---|
| avg_monthly_spend | 0.922 | Higher spend → higher risk |
| InternetService_Fiber optic | 0.785 | Fiber optic → higher risk |
| tenure | 0.774 | Longer tenure → lower risk |
| contract_risk_score | 0.521 | Month-to-month → higher risk |
| TotalCharges | 0.365 | Higher total → context-dependent |

SHAP values are computed using `LinearExplainer` (exact, no sampling). Individual customer waterfall explanations are available via the `/explain` API endpoint.

---

## Ethical Considerations

### Demographic features
`gender` and `SeniorCitizen` are included as features. Their SHAP importances are low (gender: <0.05 mean |SHAP|), but they should be audited for disparate impact before any production deployment.

**Recommended pre-deployment step:** Run disparate impact analysis segmented by gender and senior citizen status. If churn predictions correlate with protected attributes in a way that leads to differential service quality, remove those features.

### Retention offer equity
If this model is used to offer discounts to at-risk customers, audit whether high-risk customers (who get offers) systematically differ from low-risk customers (who don't) along demographic lines.

---

## Model Limitations

1. **Cross-sectional data:** No temporal signals. The model does not see whether a customer's behavior is worsening over time.
2. **No external signals:** Competitive pricing, outage history, and support tickets are not included.
3. **US telecom only:** Not validated on non-US markets or different product structures.
4. **Precision trade-off:** 63.7% of retention outreach goes to customers who would not have churned. This is acceptable given the cost matrix but should be monitored.
5. **Model drift:** Customer behavior may shift over time. Recommend retraining quarterly and monitoring precision/recall on live outcomes.

---

## Deployment Information

- **API:** FastAPI, `POST /predict` returns probability + risk tier + CLV at risk
- **Latency:** ~15ms per prediction (Logistic Regression, CPU)
- **Batch support:** `POST /predict/batch` accepts CSV files for bulk scoring
- **Explainability:** `POST /explain` returns SHAP waterfall for any single customer
- **Infrastructure:** Docker-containerized, deployable on any cloud provider

---

## Monitoring Recommendations

| Metric | Check frequency | Alert threshold |
|---|---|---|
| Churn rate in new data | Weekly | ±5pp from training baseline (26.5%) |
| Model AUC on labeled outcomes | Monthly (with 30-day lag) | < 0.78 |
| Feature distribution drift | Monthly | PSI > 0.2 on top 5 features |
| Retention campaign conversion rate | Weekly | < 10% (precision proxy) |

---

*This model card follows the Mitchell et al. (2019) model card framework.*
