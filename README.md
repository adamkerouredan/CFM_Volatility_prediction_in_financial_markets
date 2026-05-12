# CFM Volatility Forecasting Challenge

Systematic prediction of US equity intraday realized volatility (14h–16h window)
from morning intraday bars (9h30–13h55), using a rigorous quantitative research
protocol.

**Final result** : MAPE = **0.2392** on a held-out test set (15% of data),
representing a **34.9% improvement** over the official CFM benchmark.

---

## Project context

This project implements a complete pipeline for the CFM Volatility Forecasting
Challenge organized on the [Challenge Data](https://challengedata.ens.fr/) platform
by Capital Fund Management (CFM), within the *Projets Informatiques MASH*
course (Pr. Grégoire Mialon, Inria).

The methodological reference throughout the project is **Giuseppe Paleologo,
*The Elements of Quantitative Investing*, Wiley 2024** (chapters 3, 4, 5, 6, 7, 8).

---

## Final results

### Performance benchmark

| Model                              | MAPE (CV)  | MAPE (holdout) | Improvement vs baseline |
|------------------------------------|-----------:|---------------:|------------------------:|
| Baseline 1 (mean of 54 bars)       | 0.3713     | 0.3686         | reference               |
| Baseline 4 (1h mean)               | -          | 0.2921         | -20.8%                  |
| Ridge A (10 engineered features)   | 0.3090     | -              | -                       |
| HAR-RV (Corsi 2009, 3 horizons)    | 0.2732     | -              | -                       |
| LightGBM C (10 engineered)         | 0.2600     | -              | -                       |
| LightGBM D (108 raw bars)          | 0.2584     | 0.2569         | -30.3%                  |
| LightGBM E (+ product_id)          | 0.2574     | -              | -                       |
| **LightGBM F (30 structured)**     | **0.2427** | **0.2392**     | **-35.1%**              |

### Validation metrics (LightGBM F champion)

- **MAPE on holdout** : 0.2392
- **QLIKE on holdout** : 0.1928 (Patton-Sheppard 2009, Paleologo Chap. 5)
- **95% Bootstrap CI (grouped by date)** : [0.2352, 0.2451]
- **R²** : 0.78
- **Top 10% TARGET MAPE** : 0.2195 (model performs well in high-volatility regimes)
- **Pairwise test vs LightGBM D** : F dominates D statistically (CI excludes 0)

### Challenge Data ranking

Public leaderboard submission with LightGBM D: **rank 11 / 17**, score 25.74.
LightGBM F (final champion) is expected to rank approximately **6-7**.

---

## Methodology

### 1. Data and exploratory analysis
- 636,313 observations, 318 stocks, 2,117 anonymized dates
- 15.7% of rows contain at least one NaN (concentrated at 9h30 — microstructure)
- TARGET distribution: median 0.142, skewness 5.01, kurtosis 59.9
- Log-transform of TARGET (Harvey-Shephard 1996, Andersen-Benzoni 2009)
- Persistence: ρ(morning vol, afternoon vol) = 0.857

### 2. Feature engineering (Paleologo Chap. 6 — Loadings Generation)
- 10 raw features built from 54 intraday volatility bars + 54 return signs
- 10 Z-score features cross-sectional by date (robust median + MAD)
- 10 Z-score features historical by stock (fitted on training set only)
- Total: 30 structured features
- Feature selection by IC (Pearson, Kendall τ, cross-sectional Spearman)
- Multicollinearity audit by Spearman correlation

### 3. Model benchmarking
- Linear models (Ridge with winsorization, Ridge + PCA on Marchenko-Pastur)
- Academic baseline (HAR-RV with 3 nested horizons)
- Non-linear ensemble methods (LightGBM)
- Residual stacking (HAR-RV + LightGBM)
- Structured-feature LightGBM (champion)

### 4. Industrial-grade validation
- **Repeated Stratified K-Fold** (5 folds × 2 repetitions, stratified by TARGET quartile)
- **15% holdout** intouched until final evaluation
- **Bootstrap 1000× by date** (not row-by-row, respecting intra-date autocorrelation)
- **Top-vol audit** by deciles (10/5/1/0.5/0.1%)
- **Underestimation analysis** by TARGET decile (critical for risk management)
- **Pairwise comparison** between models, observation-by-observation
- **QLIKE loss** for rank-robust volatility forecast evaluation
- **Anti-leakage discipline** : all stats fitted on train only, no test-fold leakage

---

## Identified limitations

The model exhibits **systematic underestimation in high-volatility regimes**
(66% underestimation rate in TARGET decile 10), a typical shrinkage effect of
ML models trained with MSE loss in log-space. This characteristic makes the
model suitable for prediction challenges (MAPE-optimized) but would require
additional post-calibration before production use in risk management.

Documented in detail in the final report.

---

## Repository structure

```
CFM_Vol_FCT/
├── data/                          (.gitignore — provided by CFM)
├── src/
│   ├── data_loader.py             DataLoader
│   ├── eda_analyzer.py            EDAAnalyzer
│   ├── feature_engineer.py        FeatureEngineer + FeatureTransformer
│   ├── neutralizer.py             Neutralizer (date / stock)
│   ├── splitter.py                Stratified train/holdout splitter
│   ├── evaluator.py               MAPE + Jensen + residual diagnostics
│   ├── validator.py               Repeated Stratified K-Fold
│   └── model.py                   Ridge / RidgePCA / LightGBM / HAR / Stacking
├── notebooks/
│   └── main.ipynb                 Complete pipeline notebook
├── outputs/                       (.gitignore — generated figures)
├── reports/
│   └── rapport_methodologie.tex   LaTeX report
├── FICHE_PASSATION.md             Handover document
├── PIPELINE.md                    Project roadmap
└── README.md                      This file
```

---

## Usage

### Requirements

```
python      >= 3.11
numpy       >= 1.20
pandas      >= 2.0
scikit-learn >= 1.3
lightgbm    >= 4.0
matplotlib  >= 3.7
seaborn     >= 0.12
scipy       >= 1.10
```

### Run the pipeline

1. Place data files in `data/X_train/`, `data/X_test/`, `data/`
2. Open `notebooks/main.ipynb`
3. Run all cells sequentially

The notebook is structured by phase:
- Phase I — Data ingestion and EDA
- Phase II — Feature engineering
- Phase III — Modeling and validation

---

## References

- **G. Paleologo**, *The Elements of Quantitative Investing*, Wiley, 2024.
- **F. Corsi**, A Simple Approximate Long-Memory Model of Realized Volatility,
  *Journal of Financial Econometrics*, 2009.
- **A. C. Harvey, N. Shephard**, Estimation of an Asymmetric Stochastic Volatility
  Model for Asset Returns, *Journal of Business & Economic Statistics*, 1996.
- **A. J. Patton, K. Sheppard**, Evaluating Volatility and Correlation Forecasts,
  *Handbook of Financial Time Series*, Springer, 2009.
- **V. Marchenko, L. Pastur**, Distribution of Eigenvalues for Some Sets of
  Random Matrices, 1967.

---

## Author

**Adam Kerouredan** — M2 Quantitative Finance, 2025-2026.

Project completed in approximately 4 days of intensive work, May 2026.
