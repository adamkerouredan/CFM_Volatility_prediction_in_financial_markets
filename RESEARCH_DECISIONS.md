# Research Protocol — CFM Volatility Forecasting Challenge

**Objective.** Predict the realized volatility of 318 US stocks over the 2pm–4pm window,
using only intraday data observed between 9:30am and 1:55pm.
**Metric.** Mean Absolute Percentage Error (MAPE). Lower is better.
**Methodological reference.** G. Paleologo, *The Elements of Quantitative Investing*, Wiley 2024.

---

## 1. Understanding the problem

The dataset contains 636,313 observations across 318 stocks and 2,117 trading dates.
For each observation, we have 54 five-minute volatility bars (9:30am to 1:55pm),
54 return direction bars (sign of price change, values in {-1, 0, +1}), and a
target variable: the realized volatility of the same stock over the 2pm–4pm window.

Two structural constraints shaped every subsequent decision.

**Dates are anonymized.** The challenge deliberately randomizes the order of trading
days. This eliminates any approach that relies on inter-day chronological structure:
walk-forward validation in the calendar sense, GARCH-type inter-day models, and
HAR-RV with true multi-day lags are not directly applicable.

**Volatility is highly persistent.** The Pearson correlation between the morning
mean volatility and the afternoon target is 0.857. This means a naive predictor —
"the afternoon will look like the morning" — already explains 73% of the variance.
Any model we build must beat this baseline by a meaningful margin to justify its
complexity.

---

## 2. Target transformation

The target distribution is highly right-skewed: skewness 5.01, kurtosis 59.9,
minimum value 0.000132. Training a model directly on the raw target would produce
heteroscedastic residuals and poor generalization.

**Decision: log-transform the target** before modeling, following Harvey and
Shephard (1996). All models are trained on log(TARGET). Predictions are converted
back via exp() at inference time, with a Jensen correction to account for the
bias introduced by the non-linear transformation.

---

## 3. Handling missing values

15.7% of rows contain at least one NaN. The NaNs are heavily concentrated on
the 9:30am bar (28,091 cases), reflecting a well-known microstructure phenomenon:
the opening auction produces noisy or missing prices in the first five-minute window.

**For volatility bars:** linear interpolation along the intraday axis, with
forward-fill and backward-fill for edge bars. This preserves the continuity
of the intraday profile without introducing cross-row dependencies.

**For return direction bars:** imputation by zero, interpreted as the absence
of price movement. Economically neutral and avoids inventing directional
information that was not observed.

Two alternatives were rejected. Imputing by cross-sectional mean would introduce
dependencies between stocks, creating a leakage risk. Dropping rows with NaNs
would remove 15.7% of the data and introduce selection bias on market open behavior.

---

## 4. Feature engineering

Rather than feeding raw bars directly into a model, we constructed 10 features
from the 54 volatility bars and 54 return bars. Each feature was designed to
capture a specific economic dimension of the intraday profile.

### 4.1 Features retained

| Feature | Economic hypothesis |
|---|---|
| `vol_mean` | Volatility clustering: the morning average predicts the afternoon level |
| `vol_mean_recent` | The most recent bars (closest to 2pm) best reflect the current regime |
| `vol_last_bar` | The last observed bar is the most direct signal of current volatility |
| `vol_linear_slope` | The direction of the intraday trend contains predictive information |
| `vol_std` | High standard deviation signals an erratic, hard-to-predict day |
| `vol_min` | The morning floor anchors the overall volatility range |
| `vol_mean_minus_median` | The mean-median gap measures the presence of isolated spikes |
| `vol_recent_over_mean` | A ratio above 1 signals volatility accelerating toward 2pm |
| `return_n_positive` | Directional bias in returns is weakly correlated with volatility level |
| `return_n_negative` | Same rationale |

### 4.2 Features dropped after audit

Six features were initially built and removed after a formal multicollinearity
audit using Spearman correlation and Information Coefficient analysis.

| Feature dropped | Reason |
|---|---|
| `vol_median` | Pearson r = 0.969 with `vol_mean` — no additional information |
| `vol_trend` | Pearson r = 0.918 with `vol_linear_slope` — redundant |
| `vol_max` | Spearman r = 0.956 with `vol_std` — redundant |
| `vol_range` | Spearman r = 0.999 with `vol_max` — near-perfect collinearity |
| `return_direction_bias` | IC = -0.048 — no predictive signal |
| `return_last_bar` | IC = +0.002 — no predictive signal |

---

## 5. Feature evaluation

Before modeling, the 10 retained features were evaluated across three orthogonal
metrics.

**Information Coefficient (IC):** Pearson, Kendall τ, and cross-sectional Spearman
correlations computed between each feature and the log-target. All t-statistics
exceeded 35 across 2,117 dates — statistically robust signals.

**Marchenko-Pastur analysis:** The eigenvalue spectrum of the feature covariance
matrix was compared against the noise threshold (λ₊ = 1.0087). Three components
lie above the threshold (PC1 = 6.37, PC2 = 1.79, PC3 = 1.20), confirming that the
feature matrix contains genuine structure beyond random noise.

**Cross-sectional neutralization:** Removing the market-wide component from both
features and target caused the average IC to drop by approximately 13%. This means
roughly 87% of the predictive signal is idiosyncratic — stock-specific dynamics,
not a market beta proxy.

---

## 6. Feature structuring

Following Paleologo (2024), Chapter 6, each raw feature was extended into two
additional transformations, for a final matrix of 30 features.

**Raw features (10):** the absolute values computed in Section 4.

**Cross-sectional Z-scores by date (10):** for each feature and each trading day,
standardize across all stocks observed on that day, using median and MAD (scaled
by 1.4826). This answers: *is this stock more volatile than the market today?*
These Z-scores do not use the target and can be computed on the test set without
leakage.

**Historical Z-scores by stock (10):** for each feature and each stock, standardize
using that stock's mean and standard deviation computed on the training set only.
This answers: *is this stock more volatile than its own history?* Statistics are
fitted on the training set and applied identically to the test set.

---

## 7. Modeling

Eight models or configurations were tested, moving from simple to complex.
Each step was justified by a specific gap in the previous model's performance.

### 7.1 Baselines

**Baseline 1 — Mean of 54 bars** (MAPE CV = 0.3713, holdout = 0.3686).
The official CFM benchmark. Every model must beat this.

**Baseline 4 — Mean of the last hour** (holdout = 0.2921).
Using only recent bars already improves substantially, confirming the recency
effect observed in the IC analysis.

### 7.2 Linear models

**Ridge A — 10 engineered features** (MAPE CV = 0.3090).
Regularized linear regression on the engineered features. The linear reference.

**Ridge B — PCA on 3 components** (MAPE CV = 0.3365). Compressing the feature
matrix to 3 PCA components lost 24% of explained variance and degraded performance.
**Rejected:** compression discards genuine signal.

### 7.3 Academic baseline

**HAR-RV (Corsi 2009) — 3 intraday horizons** (MAPE CV = 0.2732). The
Heterogeneous Autoregressive model, implemented with intraday lags (last bar,
last 6 bars, all 54 bars) as a substitute for the unavailable inter-day lags.
Provides an academically grounded benchmark.

### 7.4 Non-linear models

**LightGBM C — 10 engineered features** (MAPE CV = 0.2600). The jump from
Ridge A (0.3090) confirms genuine non-linearities in the feature-target relationship
that a linear model cannot capture.

**LightGBM D — 108 raw bars** (MAPE CV = 0.2584, holdout = 0.2569). Feeding all
54 volatility and 54 return bars directly to LightGBM. LightGBM handles NaNs
natively, so no imputation is needed. The marginal gain over LightGBM C is small,
suggesting the 10 engineered features capture most of the information in the 108 raw
bars in a much more compact form.

**LightGBM E — 10 features + product_id** (MAPE CV = 0.2574). Adding the stock
identifier as a categorical feature. Negligible gain: LightGBM was already learning
stock-level behavior implicitly through the historical Z-score features.

**Stacking HAR-RV + LightGBM** (MAPE CV = 0.2614). A meta-learner trained on
both models' predictions. Performed worse than LightGBM alone because both models
draw from the same underlying information source. Correlated errors make
ensembling ineffective. **Rejected.**

**LightGBM F — 30 structured features** (MAPE CV = 0.2427, holdout = 0.2392).
The champion model. Adding the 20 Z-score features reduced MAPE by 6.7% relative
to LightGBM C. The cross-sectional and historical Z-scores encode information that
LightGBM was previously forced to infer implicitly — making it explicit and structured
improves both accuracy and stability.

---

## 8. Directions not explored

**Sequential models (LSTM, TCN, Transformer).** Expected gain: 1-2 MAPE points.
Excluded: significant loss of interpretability, and the project prioritized
methodological clarity over leaderboard ranking.

**GARCH and inter-day time series models.** Would require chronological order.
Excluded: dates are anonymized.

**MAPE-aware loss function** (sample_weight = 1/y²). Implemented and tested.
Global MAPE improved by ~2 points, but Q4 performance (highest-volatility quartile)
degraded from 0.21 to 0.30. **Rejected:** a risk-oriented model must perform well
precisely on high-volatility days.

**External data (VIX, market indices).** Expected gain: 1-2 points. Excluded by
methodological choice: project uses only challenge-provided data.

---

## 9. Validation protocol

### 9.1 Data split

A 15% holdout set was created before any modeling, stratified by TARGET quartile.
It was not touched until the final evaluation of the single champion model. No
hyperparameter tuning or model selection used holdout data.

### 9.2 Cross-validation

Repeated Stratified K-Fold (5 folds × 2 repetitions), stratified by TARGET quartile.
For LightGBM, early stopping used a dedicated internal validation split carved from
the training fold — never the external test fold.

### 9.3 Final evaluation

| Metric | Value |
|---|---|
| MAPE (holdout) | 0.2392 |
| QLIKE (Patton-Sheppard 2009) | 0.1928 |
| 95% bootstrap CI (grouped by date) | [0.2352, 0.2451] |
| R² | 0.78 |
| CV MAPE | 0.2427 |

The CV and holdout MAPEs are consistent and within the bootstrap CI. No overfitting detected.

**Performance by volatility regime:**

| Quartile | TARGET range | MAPE | Note |
|---|---|---|---|
| Q1 (calm) | < 0.10 | 0.3573 | MAPE artifact on small denominators |
| Q2 | 0.10–0.14 | 0.2017 | Strong |
| Q3 | 0.14–0.21 | 0.1897 | Best absolute performance |
| Q4 (volatile) | > 0.21 | 0.2083 | Strong — the regime that matters for risk |
| Top 10% | > 0.33 | 0.2195 | Better than global average |

The model performs better in high-volatility regimes than on average — the correct
property for a risk management application.

---

## 10. Known limitations

**Systematic underestimation in high-volatility regimes.** The model underestimates
66% of observations in the highest decile. This is a known shrinkage artifact of
ML models trained with MSE loss in log-space. In production, this would require
correction via multiplicative recalibration by decile or asymmetric loss functions.

**Degradation at extreme tails.** MAPE reaches 29.7% at the top 0.1% of the target
distribution. An Extreme Value Theory overlay could improve this behavior.

**MAPE dominated by quiet days.** The 24% global MAPE is driven upward by Q1
(35.7%), while Q2-Q4 perform at approximately 20%. This is a property of the
metric, not of the model.

---

## 11. Summary

| Decision point | Choice made | Rejected alternatives |
|---|---|---|
| Target | Log-transform | Raw level |
| NaN volatility | Intraday interpolation | Cross-sectional mean, row deletion |
| NaN returns | Fill with zero | Interpolation |
| Features | 10 hand-crafted | Raw bars only |
| Feature structuring | Raw + Z-date + Z-stock (30 total) | Raw only, PCA compression |
| Linear model | Ridge on raw features | Ridge on PCA |
| Non-linear model | LightGBM on 30 structured features | LSTM, stacking |
| Loss function | MSE in log-space | MAPE-aware weighting |
| Validation | Repeated Stratified K-Fold + stratified holdout | Standard K-Fold |

**Final result: MAPE = 24.25 on the public leaderboard — rank 8 / 17.**
CFM official benchmark: 68.14. Improvement factor: 2.8×.

---

*Adam Kerouredan — M2 Quantitative Finance, 2025-2026.*
*Reference: G. Paleologo, The Elements of Quantitative Investing, Wiley 2024.*
