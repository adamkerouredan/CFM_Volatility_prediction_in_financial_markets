# CFM Volatility Forecasting Challenge

Systematic prediction of US equity intraday realized volatility over the 14:00–16:00 window from intraday information observed between 09:30 and 13:55.

The project follows a conservative quantitative research protocol: explicit baselines, engineered volatility features, train-only transformations, holdout validation, date-grouped bootstrap, regime analysis, and robustness checks.

**Official model retained:** LightGBM F — 30 structured features, `n_estimators=2000`.

**Internal holdout result:** MAPE = **0.2398** on a 15% untouched holdout.

**Public Challenge Data result:** MAPE = **24.25**, rank **8 / 17**.

> Note: a post-holdout diagnostic with `n_estimators=3000` reached MAPE = **0.2392**, but it is not retained as the official model because it was tested after observing the holdout.

---

## Project context

This project implements a complete pipeline for the CFM Volatility Forecasting Challenge organized on the Challenge Data platform by Capital Fund Management (CFM), within the *Projets Informatiques MASH* course.

The methodological reference throughout the project is Giuseppe Paleologo, *The Elements of Quantitative Investing*, Wiley 2024, especially the chapters on signal construction, model validation, risk-aware evaluation, and portfolio-style research discipline.

---

## Dataset

- **636,313** training observations.
- **635,397** test observations.
- **318** stocks.
- **2,117** anonymized dates.
- **54** intraday volatility bars from 09:30 to 13:55.
- **54** return-direction bars over the same intraday window.
- Target: realized volatility of the same stock over 14:00–16:00.
- Official metric: **MAPE**.

The target is strongly right-skewed, with skewness around **5.01** and kurtosis around **59.9**. The morning volatility profile is highly persistent: the correlation between morning volatility and the afternoon target is approximately **0.857**.

---

## Final performance

### Holdout apples-to-apples comparison

| Model | MAPE holdout | Gain vs Baseline 1 | Gain vs Baseline 4 |
|---|---:|---:|---:|
| **LightGBM F — 30 structured features** | **0.2398** | **34.92%** | **17.87%** |
| LightGBM D — 108 raw bars | 0.2581 | 29.98% | 11.64% |
| HAR-RV — 3 horizons | 0.2728 | 25.98% | 6.59% |
| Baseline 4 — mean of last hour | 0.2921 | 20.76% | 0.00% |
| Ridge A — 10 engineered features | 0.3085 | 16.30% | -5.63% |
| Baseline 3 — mean of last 30 minutes | 0.3340 | 9.38% | -14.36% |
| Baseline 1 — mean of 54 bars | 0.3686 | 0.00% | -26.20% |
| Baseline 2 — last bar | 0.5662 | -53.63% | -93.87% |

### Champion validation metrics

| Metric | Value |
|---|---:|
| CV MAPE | 0.2427 |
| Holdout MAPE | 0.2398 |
| 95% bootstrap CI, grouped by date | [0.2352, 0.2451] |
| QLIKE holdout | 0.1941 |
| R² holdout | 0.7986 |
| Pairwise MAPE gap vs LightGBM D | +0.0182 |
| Pairwise bootstrap CI on MAPE(D) - MAPE(F) | [+0.0159, +0.0209] |

LightGBM F is stable across cross-validation and holdout. The gain is not a one-off split artifact: the pairwise bootstrap against LightGBM D remains strictly positive.

---

## Methodology

### 1. Data ingestion and EDA

The project first audits the raw data shape, missing values, target distribution, and intraday volatility profile.

Key findings:

- Missing values are concentrated at the first intraday bar, consistent with market-open microstructure noise.
- The target is highly skewed, which motivates training in log-space.
- The intraday volatility profile has the expected decreasing shape after the open.
- Morning volatility is strongly persistent into the afternoon.

### 2. Feature engineering

The final raw feature set contains 10 interpretable features:

| Feature | Economic interpretation |
|---|---|
| `vol_mean` | Absolute morning volatility level |
| `vol_mean_recent` | Recent volatility regime close to 14:00 |
| `vol_last_bar` | Last observed volatility state |
| `vol_linear_slope` | Intraday acceleration or compression |
| `vol_std` | Instability of the morning volatility path |
| `vol_min` | Morning volatility floor |
| `vol_mean_minus_median` | Asymmetry / isolated spikes |
| `vol_recent_over_mean` | Recent-volatility acceleration ratio |
| `return_n_positive` | Directional activity proxy |
| `return_n_negative` | Directional activity proxy |

The 10 raw features are then expanded into **30 structured features**:

1. **10 raw features**.
2. **10 cross-sectional Z-scores by date**, using robust median and MAD.
3. **10 historical Z-scores by stock**, fitted on the training set only.

The economic idea is simple: the model should know not only the absolute volatility level, but also whether a stock is unusually volatile relative to the market on the same day and relative to its own history.

### 3. Feature diagnostics

Feature evaluation combines three diagnostics:

- Information Coefficient against `log(TARGET)`.
- Cross-sectional Spearman IC by anonymized date.
- Marchenko-Pastur analysis of the feature covariance spectrum.

Updated Marchenko-Pastur result:

| Component | Eigenvalue | Interpretation |
|---|---:|---|
| PC1 | 4.7474 | Signal |
| PC2 | 1.6811 | Signal |
| PC3 | 1.1637 | Signal |
| PC4 and below | below λ+ ≈ 1.0079 | Noise-dominated |

Only three components clearly stand above the random-matrix noise threshold, but PCA compression underperforms the raw structured representation. The retained conclusion is therefore: PCA is useful diagnostically, but not retained as the production representation.

### 4. Models benchmarked

The notebook moves from simple to more expressive models:

- Baseline 1: mean of 54 morning volatility bars.
- Baseline 2: last observed volatility bar.
- Baseline 3: mean of the last 30 minutes.
- Baseline 4: mean of the last hour.
- Ridge A: linear model on 10 engineered features.
- Ridge B: Ridge on PCA components.
- HAR-RV: intraday approximation of Corsi-style heterogeneous volatility horizons.
- LightGBM C: 10 engineered features.
- LightGBM D: 108 raw intraday bars.
- LightGBM E: engineered features plus product identifier.
- Stacking HAR-RV + LightGBM.
- LightGBM F: 30 structured features.

The retained champion is **LightGBM F**, because it beats the raw-bar LightGBM while remaining economically interpretable.

---

## Robustness analysis

### Feature-family ablation

| Representation | Number of features | MAPE holdout | Gain vs Baseline 4 | Gap vs Full F |
|---|---:|---:|---:|---:|
| Raw only | 10 | 0.2569 | 12.05% | 0.0170 |
| Raw + Z-date | 20 | 0.2420 | 17.14% | 0.0022 |
| Raw + Z-stock | 20 | 0.2532 | 13.31% | 0.0133 |
| Raw + Z-date + Z-stock | 30 | 0.2398 | 17.87% | 0.0000 |

The main incremental gain comes from the cross-sectional Z-date features. This is economically intuitive: in volatility forecasting, it matters whether a stock is volatile in absolute terms, but also whether it is unusually volatile relative to the market on that same date.

### Regime analysis by TARGET decile

LightGBM F underperforms Baseline 4 in the very quiet regime, where MAPE is unstable because the denominator is small. From D3 onward, it dominates strongly.

| Regime | Interpretation |
|---|---|
| D1–D2 | Weak or negative gain due to MAPE sensitivity on tiny targets |
| D3–D10 | Strong positive gain across economically meaningful volatility regimes |

This is a key practical result: the model is most valuable when volatility is large enough to matter economically.

### Stability by anonymized date blocks

LightGBM F beats Baseline 4 in all five anonymized date blocks, with gains around 16%–21%. This suggests the result is not concentrated in a single holdout sub-period.

### Candidate features

An additional candidate feature pack improves CV MAPE from **0.2427** to **0.2421**, but the gain is small relative to fold-level variability. These features are promising but not retained in the official submission model.

### More regularized LightGBM

A more regularized LightGBM obtains CV MAPE **0.2438**, close to the champion but slightly worse. This supports the view that the champion is not a fragile overfit, while still justifying keeping the official LightGBM F specification.

---

## Identified limitations

1. **Very quiet days:** MAPE is unstable for very small volatility targets. The model underperforms Baseline 4 in the lowest target decile.
2. **Extreme tails:** performance deteriorates in the top 0.5% and top 0.1% volatility regimes. These regimes would require dedicated tail modeling before production risk usage.
3. **Underestimation risk:** the model underestimates more frequently as the target decile increases. This is a common shrinkage effect for ML models trained in log-space.
4. **No true calendar walk-forward:** dates are anonymized, so temporal validation is necessarily based on anonymized blocks rather than real chronological market regimes.
5. **Candidate features not retained:** small CV improvement is not enough to justify changing the official model without additional validation.

---

## Repository structure

```text
CFM_Vol_FCT/
├── data/                          (.gitignore — provided by CFM)
├── src/
│   ├── data_loader.py             DataLoader
│   ├── eda_analyzer.py            EDAAnalyzer
│   ├── feature_engineer.py        FeatureEngineer + FeatureTransformer
│   ├── neutralizer.py             Neutralizer
│   ├── splitter.py                Stratified train/holdout splitter
│   ├── evaluator.py               MAPE + Jensen + residual diagnostics
│   ├── validator.py               Repeated Stratified K-Fold
│   └── model.py                   Ridge / RidgePCA / LightGBM / HAR / Stacking
├── notebooks/
│   └── main.ipynb                 Complete pipeline notebook
├── outputs/                       (.gitignore — regenerated by notebook)
├── PIPELINE.md
├── RESEARCH_DECISIONS.md
└── README.md
```

---

## Usage

```bash
python -m pip install -r requirements.txt
cd notebooks
jupyter nbconvert --to notebook --execute main.ipynb --inplace
```

Expected generated artifacts include:

```text
outputs/submission.csv
outputs/robustness_holdout_apples_to_apples.csv
outputs/feature_family_ablation_holdout.csv
outputs/regime_decile_f_vs_baseline4.csv
outputs/pseudo_time_blocks_f_vs_baseline4.csv
outputs/candidate_features_cv.csv
outputs/regularized_lgbm_cv.csv
```

---

## References

- Giuseppe Paleologo, *The Elements of Quantitative Investing*, Wiley, 2024.
- F. Corsi, “A Simple Approximate Long-Memory Model of Realized Volatility,” *Journal of Financial Econometrics*, 2009.
- A. C. Harvey and N. Shephard, “Estimation of an Asymmetric Stochastic Volatility Model for Asset Returns,” 1996.
- A. J. Patton and K. Sheppard, “Evaluating Volatility and Correlation Forecasts,” 2009.
- V. Marchenko and L. Pastur, “Distribution of Eigenvalues for Some Sets of Random Matrices,” 1967.
- M. López de Prado, *Advances in Financial Machine Learning*, Wiley, 2018.

---

## Author

Adam Kerouredan — M2 Quantitative Finance, 2025–2026.
