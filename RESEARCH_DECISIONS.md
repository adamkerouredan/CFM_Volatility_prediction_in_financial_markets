# Research Protocol — CFM Volatility Forecasting Challenge

**Objective.** Predict the realized volatility of US equities over the 14:00–16:00 window using only intraday data observed between 09:30 and 13:55.

**Metric.** Mean Absolute Percentage Error (MAPE). Lower is better.

**Methodological reference.** Giuseppe Paleologo, *The Elements of Quantitative Investing*, Wiley 2024.

---

## 1. Problem understanding

The dataset contains **636,313** training observations across **318** stocks and **2,117** anonymized dates. For each observation, we observe:

- 54 five-minute volatility bars from 09:30 to 13:55;
- 54 return-direction bars over the same window;
- one target variable: realized volatility of the same stock from 14:00 to 16:00.

Two constraints drive the research design.

### 1.1 Dates are anonymized

The challenge anonymizes and randomizes trading dates. This makes true chronological modeling impossible: calendar walk-forward validation, inter-day GARCH, macro regime features, and true multi-day HAR lags are not directly available.

The validation framework therefore uses stratified holdout, repeated stratified cross-validation, date-grouped bootstrap, and anonymized date-block stability checks.

### 1.2 Volatility persistence is strong

The correlation between morning volatility and afternoon realized volatility is approximately **0.857**. This makes naive volatility persistence a strong benchmark.

A model is only useful if it beats simple baselines by a meaningful and robust margin.

---

## 2. Target transformation

The target is strongly right-skewed, with skewness around **5.01** and kurtosis around **59.9**.

**Decision.** Train models on `log(TARGET)`.

**Rationale.**

- Raw volatility is positive and highly skewed.
- Log-space stabilizes residuals.
- Volatility models are naturally multiplicative.
- Jensen correction can be applied when converting predictions back to raw space.

**Risk.** Log-space training can create shrinkage in high-volatility regimes. This is later audited via underestimation-by-decile and QLIKE.

---

## 3. Missing values

15.7% of rows contain at least one missing value, with missingness concentrated near the market open.

### 3.1 Volatility bars

**Decision.** Linear interpolation along the intraday axis, followed by forward-fill and backward-fill.

**Rationale.** Missing volatility bars are an intraday continuity issue, not a cross-sectional information problem. Interpolation preserves the shape of each stock-day path without borrowing information from other stocks.

### 3.2 Return-direction bars

**Decision.** Fill missing return-direction values with zero.

**Rationale.** Zero is interpreted as absence of observed directional movement. It avoids creating artificial directionality.

### 3.3 Alternatives rejected

| Alternative | Reason rejected |
|---|---|
| Row deletion | Would remove 15.7% of rows and bias the sample toward cleaner market-open days |
| Cross-sectional mean imputation | Would introduce dependencies across stocks |
| Random imputation | Not economically interpretable |

---

## 4. Feature engineering

The initial raw input contains 108 intraday columns. Instead of using raw bars only, the project builds interpretable features that summarize economically meaningful aspects of the morning volatility path.

### 4.1 Retained raw features

| Feature | Economic hypothesis |
|---|---|
| `vol_mean` | Morning volatility level is persistent into the afternoon |
| `vol_mean_recent` | Recent bars are closer to the forecast window |
| `vol_last_bar` | Last observed state is a direct regime proxy |
| `vol_linear_slope` | Intraday compression or acceleration matters |
| `vol_std` | Volatility-of-volatility signals unstable regimes |
| `vol_min` | The observed volatility floor anchors the day |
| `vol_mean_minus_median` | Mean-median gap captures isolated spikes |
| `vol_recent_over_mean` | Ratio above 1 indicates acceleration toward 14:00 |
| `return_n_positive` | Positive directional activity can proxy market activity |
| `return_n_negative` | Negative directional activity can proxy market stress |

### 4.2 Features dropped after audit

| Feature dropped | Reason |
|---|---|
| `vol_median` | Redundant with `vol_mean` |
| `vol_trend` | Redundant with `vol_linear_slope` |
| `vol_max` | Redundant with `vol_std` |
| `vol_range` | Redundant with `vol_max` |
| `return_direction_bias` | Weak IC |
| `return_last_bar` | Near-zero IC |

The retained design favors a compact feature set with direct economic meaning.

---

## 5. Feature evaluation

Three diagnostics are used before modeling.

### 5.1 Information Coefficient

The strongest features are volatility-level and volatility-shape variables. `vol_mean`, `vol_mean_recent`, and `vol_std` dominate, which is consistent with volatility clustering and intraday persistence.

### 5.2 Marchenko-Pastur analysis

The covariance eigenvalue spectrum shows three components above the random-matrix noise threshold:

| Component | Eigenvalue | Interpretation |
|---|---:|---|
| PC1 | 4.7474 | Signal |
| PC2 | 1.6811 | Signal |
| PC3 | 1.1637 | Signal |
| PC4+ | below λ+ ≈ 1.0079 | Noise-dominated |

**Decision.** PCA is useful as a diagnostic of low-dimensional structure, but PCA-based Ridge is not retained because it underperforms. The compression discards information useful for prediction.

### 5.3 Cross-sectional neutralization

After cross-sectional neutralization, the average IC drops by roughly **20–22%**. This means that part of the signal is market-wide, but a substantial idiosyncratic component remains.

**Interpretation.** The model is not merely predicting market beta. Stock-specific morning volatility dynamics still contain useful signal.

---

## 6. Feature structuring

The final representation expands the 10 raw features into 30 structured features.

| Family | Number | Question answered |
|---|---:|---|
| Raw features | 10 | What is the absolute volatility state? |
| Z-date | 10 | Is this stock unusually volatile relative to other stocks today? |
| Z-stock | 10 | Is this stock unusually volatile relative to its own history? |

### 6.1 Z-date

Computed cross-sectionally by anonymized date using robust median and MAD.

**Leakage status.** No target is used. Test-set Z-date features can be computed because all same-date stock observations are available at prediction time in the challenge setting.

### 6.2 Z-stock

Computed using stock-level training statistics and then applied to holdout/test.

**Leakage status.** Training statistics only. No validation or test labels are used.

---

## 7. Modeling decisions

The modeling path deliberately moves from simple to complex.

### 7.1 Baselines

**Baseline 1 — Mean of 54 bars.**  
MAPE CV = 0.3713, holdout = 0.3686. This is the main official-style persistence benchmark.

**Baseline 4 — Mean of last hour.**  
Holdout = 0.2921. This is a very strong naive benchmark. It confirms that recency matters: volatility close to 14:00 is more informative than the whole morning average.

### 7.2 Linear models

**Ridge A — 10 engineered features.**  
CV MAPE = 0.3090, holdout = 0.3085. It beats Baseline 1 but not Baseline 4 on holdout. The linear model is useful as an interpretable reference, not as champion.

**Ridge B — PCA 3 components.**  
CV MAPE = 0.3365. Rejected: PCA compression removes predictive signal even though Marchenko-Pastur shows real structure.

### 7.3 Academic volatility benchmark

**HAR-RV — 3 intraday horizons.**  
CV MAPE = 0.2732, holdout = 0.2728. It provides a strong economically grounded reference.

Since true inter-day lags are unavailable, HAR-RV is implemented with intraday horizons: recent, medium, and full-morning volatility.

### 7.4 Non-linear models

**LightGBM C — 10 engineered features.**  
CV MAPE = 0.2600. The gain over Ridge A confirms non-linearities in the feature-target relation.

**LightGBM D — 108 raw bars.**  
CV MAPE = 0.2584, holdout = 0.2581. This is a strong raw-data benchmark. It confirms that raw bars contain signal, but not enough to beat the structured representation.

**LightGBM E — product identifier added.**  
CV MAPE = 0.2574. The gain is marginal. Product identity alone is not enough to justify making the model less clean.

**Stacking HAR-RV + LightGBM.**  
CV MAPE = 0.2614. Rejected because the errors are likely correlated; both models exploit the same volatility persistence source.

**LightGBM F — 30 structured features.**  
CV MAPE = **0.2427**, official holdout MAPE = **0.2398**. Retained as champion.

### 7.5 Post-holdout diagnostic

A diagnostic run with `n_estimators=3000` achieved MAPE = **0.2392**.

**Decision.** Not retained as official champion because it was tested after observing holdout performance. The retained model remains LightGBM F with `n_estimators=2000`.

---

## 8. Validation protocol

### 8.1 Holdout

A 15% holdout set is created and kept untouched until final evaluation.

### 8.2 Cross-validation

Repeated Stratified K-Fold with 5 folds × 2 repetitions, stratified by TARGET quartile.

### 8.3 Early stopping

LightGBM early stopping uses an internal validation split carved from the training fold, never the external validation fold or holdout.

### 8.4 Bootstrap

Uncertainty is measured by **date-grouped bootstrap**, not row-by-row bootstrap. This respects intra-date dependence across stocks.

### 8.5 QLIKE

QLIKE is used as an auxiliary volatility-forecasting loss. It penalizes poorly calibrated variance forecasts, especially underprediction.

---

## 9. Final evaluation

| Metric | Value |
|---|---:|
| CV MAPE | 0.2427 |
| Holdout MAPE | 0.2398 |
| Date-grouped bootstrap 95% CI | [0.2352, 0.2451] |
| QLIKE holdout | 0.1941 |
| R² holdout | 0.7986 |
| Public leaderboard MAPE | 24.25 |
| Public rank | 8 / 17 |

The CV and holdout results are consistent. The public score is close to the internal validation level.

---

## 10. Robustness results

### 10.1 Apples-to-apples holdout table

LightGBM F dominates all relevant benchmarks on the same holdout set:

| Model | MAPE holdout |
|---|---:|
| LightGBM F — 30 structured features | 0.2398 |
| LightGBM D — raw bars | 0.2581 |
| HAR-RV | 0.2728 |
| Baseline 4 | 0.2921 |
| Ridge A | 0.3085 |
| Baseline 1 | 0.3686 |

### 10.2 Ablation

| Representation | MAPE holdout | Interpretation |
|---|---:|---|
| Raw only | 0.2569 | Strong, but incomplete |
| Raw + Z-date | 0.2420 | Main incremental gain |
| Raw + Z-stock | 0.2532 | Positive but smaller |
| Full F | 0.2398 | Best representation |

**Decision.** Keep the full 30-feature representation. The ablation shows that Z-date is the main driver of the improvement.

### 10.3 Volatility regimes

LightGBM F underperforms Baseline 4 in D1 and is roughly neutral in D2. From D3 to D10, it delivers strong positive gains.

**Interpretation.** The model is most valuable in economically meaningful volatility regimes. The poorest relative behavior occurs where MAPE is least stable: tiny targets.

### 10.4 Date-block robustness

LightGBM F beats Baseline 4 in every anonymized date block, with gains around 16%–21%.

**Interpretation.** The result is not concentrated in one holdout segment.

### 10.5 Candidate features

Candidate economic features improve CV MAPE from 0.2427 to 0.2421.

**Decision.** Not retained. The improvement is too small relative to CV noise and would require additional out-of-sample confirmation.

### 10.6 Regularized LightGBM

A more regularized LightGBM obtains CV MAPE 0.2438.

**Decision.** Not retained. The result is close enough to confirm robustness, but not better than the official champion.

---

## 11. Known limitations

### 11.1 Quiet-day MAPE instability

D1-D2 are problematic because MAPE over-penalizes small denominators. This is a metric limitation, not necessarily a large economic error.

### 11.2 Extreme-tail degradation

Performance deteriorates in the top 0.5% and top 0.1% of the target distribution. A production-grade risk model would need tail-specific calibration.

### 11.3 Underestimation in high-volatility regimes

Underestimation frequency rises with target decile. This is consistent with shrinkage in log-space ML models.

### 11.4 No real chronological walk-forward

Because dates are anonymized, the project cannot verify robustness across real market regimes such as crisis, rebound, or low-volatility periods.

---

## 12. Rejected alternatives

| Alternative | Status | Reason |
|---|---|---|
| Raw target modeling | Rejected | Target skewness and heteroscedasticity |
| PCA representation | Rejected | Worse than 10 raw engineered features |
| Stacking | Rejected | Correlated errors, worse than LightGBM F |
| Product ID categorical | Rejected as champion | Marginal gain, less clean |
| MAPE-aware weighting | Rejected | Degraded high-volatility regime behavior |
| Sequential models | Not explored | Less interpretable, not needed for robust baseline |
| External data | Not used | Kept challenge-input-only discipline |
| Candidate features | Not retained | Positive but marginal CV gain |
| More regularized LightGBM | Not retained | Robust but slightly worse |

---

## 13. Final decision

The final retained model is:

```text
LightGBM F
Input: 30 structured features
Target: log(TARGET)
n_estimators: 2000
Validation: repeated stratified CV + untouched 15% holdout
Submission: outputs/submission.csv
```

Final internal holdout MAPE: **0.2398**.  
Public leaderboard score: **24.25**.  
Public rank: **8 / 17**.

---

*Adam Kerouredan — M2 Quantitative Finance, 2025–2026.*  
*Reference: Giuseppe Paleologo, The Elements of Quantitative Investing, Wiley 2024.*
