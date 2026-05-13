# PIPELINE COMPLET — CFM VOLATILITY FORECASTING CHALLENGE

> **Objectif** : prédire la volatilité réalisée d'actions américaines sur la fenêtre 14h–16h à partir des barres intraday observées entre 9h30 et 13h55.  
> **Métrique officielle** : MAPE.  
> **Référence méthodologique** : Giuseppe Paleologo, *The Elements of Quantitative Investing*, Wiley 2024.

---

## STATUT GLOBAL

| Phase | Statut | Description |
|---|---|---|
| Phase 0 | ✅ Terminé | Setup global, chemins, constantes, chargement des données |
| Phase I | ✅ Terminé | Ingestion, EDA et audit des NaN |
| Phase II | ✅ Terminé | Feature engineering : 10 features économiques |
| Phase III | ✅ Terminé | IC, neutralisation, Marchenko-Pastur |
| Phase IV | ✅ Terminé | Baselines naïves et modèles linéaires |
| Phase V | ✅ Terminé | HAR-RV et LightGBM variants |
| Phase VI | ✅ Terminé | Validation industrielle du champion |
| Phase VII | ✅ Terminé | Diagnostics risk-management et QLIKE |
| Phase VIII | ✅ Terminé | Visualisations et interprétation |
| Phase IX | ✅ Terminé | Tests de robustesse et pistes d'amélioration |
| Phase X | ✅ Terminé | Synthèse finale |
| Phase XI | ✅ Terminé | Génération de `outputs/submission.csv` |

---

## ARCHITECTURE DU PROJET

```text
CFM_Vol_FCT/
├── data/
│   ├── X_train/training_input.csv      (gitignore)
│   ├── X_test/testing_input.csv        (gitignore)
│   └── Y_train.csv                     (gitignore)
├── src/
│   ├── data_loader.py                  ✅
│   ├── eda_analyzer.py                 ✅
│   ├── feature_engineer.py             ✅
│   ├── neutralizer.py                  ✅
│   ├── splitter.py                     ✅
│   ├── validator.py                    ✅
│   ├── evaluator.py                    ✅
│   └── model.py                        ✅
├── notebooks/
│   └── main.ipynb                      ✅
├── outputs/                            (gitignore)
├── PIPELINE.md
├── RESEARCH_DECISIONS.md
└── README.md
```

---

## PHASE I — INGESTION & EDA ✅

### Résultats clés

- **636 313 observations** d'entraînement.
- **635 397 observations** de test.
- **318 stocks** et **2 117 dates anonymisées**.
- **54 barres de volatilité** entre 09h30 et 13h55.
- **54 barres de direction de rendement** sur la même fenêtre.
- **15.7%** des lignes d'entraînement contiennent au moins un NaN.
- NaN concentrés à 09h30 : signal de microstructure à l'ouverture.
- TARGET fortement asymétrique : skewness ≈ **5.01**, kurtosis ≈ **59.9**.
- Corrélation entre volatilité moyenne du matin et TARGET : **0.857**.

### Décisions

- ✅ Imputation des barres de volatilité par interpolation intraday + ffill + bfill.
- ✅ Imputation des barres de direction de rendement par zéro.
- ✅ Modélisation de `log(TARGET)` pour stabiliser la distribution.
- ✅ Pas de dépendance à l'ordre calendaire réel, car les dates sont anonymisées par CFM.

---

## PHASE II — FEATURE ENGINEERING ✅

### Feature set retenu — 10 features brutes

| Feature | IC Pearson | Kendall τ | IC cross-sectionnel | Lecture économique |
|---|---:|---:|---:|---|
| `vol_mean` | +0.7682 | +0.6287 | +0.7470 | Niveau absolu de volatilité du matin |
| `vol_mean_recent` | +0.6777 | +0.5518 | +0.6217 | Régime récent juste avant 14h |
| `vol_std` | +0.6533 | +0.5170 | +0.5860 | Instabilité de la trajectoire intraday |
| `vol_min` | +0.5621 | +0.3372 | +0.4071 | Plancher de volatilité observé |
| `vol_last_bar` | +0.5342 | +0.3678 | +0.3912 | Dernier état observable |
| `vol_mean_minus_median` | +0.5191 | +0.3571 | +0.3707 | Asymétrie / spikes intraday |
| `vol_linear_slope` | -0.4211 | -0.2707 | -0.3620 | Compression ou accélération intraday |
| `return_n_negative` | +0.2116 | +0.1359 | +0.1621 | Activité directionnelle négative |
| `vol_recent_over_mean` | +0.1918 | +0.1354 | +0.0899 | Accélération relative de la vol récente |
| `return_n_positive` | +0.1490 | +0.0885 | +0.1390 | Activité directionnelle positive |

### Features supprimées

| Feature supprimée | Raison |
|---|---|
| `vol_median` | Très redondante avec `vol_mean` |
| `vol_trend` | Redondante avec `vol_linear_slope` |
| `vol_max` | Très corrélée à `vol_std` |
| `vol_range` | Très corrélée à `vol_max` |
| `return_direction_bias` | Signal faible |
| `return_last_bar` | Signal quasi nul |

### Marchenko-Pastur

- Nombre de features : **10**.
- Nombre d'observations : **636 313**.
- λ+ ≈ **1.0079**.
- Composantes au-dessus du bruit :
  - PC1 = **4.7474**
  - PC2 = **1.6811**
  - PC3 = **1.1637**

Lecture : il existe une structure factorielle réelle, mais la compression PCA dégrade ensuite la performance prédictive. PCA est conservé comme diagnostic, pas comme représentation finale.

---

## PHASE III — STRUCTURATION DES FEATURES ✅

Le modèle champion utilise **30 features structurées** :

| Famille | Nombre | Description |
|---|---:|---|
| Features brutes | 10 | Niveau absolu de volatilité et structure intraday |
| Z-date | 10 | Positionnement cross-sectionnel du stock le même jour |
| Z-stock | 10 | Positionnement du stock par rapport à son historique d'entraînement |

La neutralisation cross-sectionnelle réduit l'IC moyen d'environ **20–22%**. Le signal n'est donc pas uniquement un proxy de marché : il contient encore une composante idiosyncratique exploitable.

---

## PHASE IV — MODÉLISATION ✅

### Résultats principaux

| Modèle | MAPE CV | MAPE holdout | Statut |
|---|---:|---:|---|
| Baseline 1 — moyenne 54 barres | 0.3713 | 0.3686 | Référence officielle |
| Baseline 4 — moyenne 1h | - | 0.2921 | Baseline naïve forte |
| Ridge A — 10 features | 0.3090 | 0.3085 | Modèle linéaire interprétable |
| Ridge B — PCA 3 composantes | 0.3365 | - | Rejeté : compression coûteuse |
| HAR-RV — 3 horizons | 0.2732 | 0.2728 | Benchmark économétrique |
| LightGBM C — 10 features | 0.2600 | - | Non-linéarité utile |
| LightGBM D — 108 barres brutes | 0.2584 | 0.2581 | Référence brute forte |
| LightGBM E — + product_id | 0.2574 | - | Gain marginal |
| Stacking HAR + LightGBM | 0.2614 | - | Rejeté |
| **LightGBM F — 30 features structurées** | **0.2427** | **0.2398** | **Champion officiel** |

### Diagnostic exploratoire non retenu

- LightGBM F avec `n_estimators=3000` : MAPE holdout = **0.2392**.
- Ce résultat est classé comme **diagnostic post-holdout**.
- Le modèle officiel reste **LightGBM F, n_estimators=2000**, pour préserver la discipline de validation.

---

## PHASE V — VALIDATION INDUSTRIELLE ✅

### Champion officiel

- Modèle : **LightGBM F**.
- Features : **30 features structurées**.
- Validation : Repeated Stratified K-Fold, 5 folds × 2 répétitions.
- Holdout : 15% stratifié par quartile de TARGET.
- Bootstrap : 1000× groupé par date.
- Correction Jensen : variance résiduelle issue de la CV.

### Métriques holdout

| Métrique | Valeur |
|---|---:|
| MAPE holdout | **0.2398** |
| IC bootstrap 95% | [0.2352, 0.2451] |
| QLIKE | 0.1941 |
| R² | 0.7986 |
| MAPE LightGBM D | 0.2581 |
| Gain pairwise F vs D | +0.0182 |

Le bootstrap pairwise par date donne un intervalle strictement positif pour `MAPE_D - MAPE_F`, ce qui confirme que F domine D statistiquement.

---

## PHASE VI — ROBUSTESSE ✅

### Table holdout apples-to-apples

| Modèle | MAPE holdout | Gain vs Baseline 1 | Gain vs Baseline 4 |
|---|---:|---:|---:|
| LightGBM F — 30 features structurées | 0.2398 | 34.92% | 17.87% |
| LightGBM D — 108 barres brutes | 0.2581 | 29.98% | 11.64% |
| HAR-RV — 3 horizons | 0.2728 | 25.98% | 6.59% |
| Baseline 4 — moyenne 1h | 0.2921 | 20.76% | 0.00% |
| Ridge A — 10 features | 0.3085 | 16.30% | -5.63% |
| Baseline 3 — moyenne 30 minutes | 0.3340 | 9.38% | -14.36% |
| Baseline 1 — moyenne 54 barres | 0.3686 | 0.00% | -26.20% |
| Baseline 2 — dernière barre | 0.5662 | -53.63% | -93.87% |

### Ablation par familles de features

| Représentation | Nb features | MAPE holdout | Gain vs Baseline 4 | Écart vs Full F |
|---|---:|---:|---:|---:|
| Brut seul | 10 | 0.2569 | 12.05% | 0.0170 |
| Brut + Z-date | 20 | 0.2420 | 17.14% | 0.0022 |
| Brut + Z-stock | 20 | 0.2532 | 13.31% | 0.0133 |
| Brut + Z-date + Z-stock | 30 | 0.2398 | 17.87% | 0.0000 |

Conclusion : le principal gain additionnel vient du **Z-date**, donc du positionnement relatif du stock dans le marché le même jour.

### Régimes de volatilité

Le modèle est faible sur les déciles très calmes D1-D2, où le MAPE est mécaniquement instable. Il domine Baseline 4 de D3 à D10, avec un gain particulièrement fort dans les régimes économiquement significatifs.

### Blocs de dates anonymisées

LightGBM F bat Baseline 4 dans chacun des 5 blocs de dates anonymisées, avec des gains proches de **16–21%**. Le résultat n'est donc pas concentré sur une seule sous-période du holdout.

### Features candidates

- LightGBM F officiel : CV MAPE = **0.2427**.
- LightGBM F + features candidates : CV MAPE = **0.2421**.

Le gain est positif mais marginal relativement à la variabilité de CV. Les features candidates sont prometteuses, mais non retenues dans le champion officiel.

### LightGBM plus régularisé

- LightGBM F officiel : CV MAPE = **0.2427**.
- LightGBM régularisé : CV MAPE = **0.2438**.

Le modèle régularisé reste proche du champion. Cela renforce la robustesse du signal, mais ne justifie pas de remplacer le champion officiel.

---

## PHASE VII — LIMITATIONS ✅

- **Jours très calmes** : D1-D2 restent difficiles à cause de la structure du MAPE.
- **Queues extrêmes** : dégradation en top 0.5% et top 0.1%.
- **Underestimation** : la probabilité de sous-estimation augmente avec le décile de TARGET.
- **Dates anonymisées** : pas de vrai walk-forward calendaire possible.
- **Features candidates** : gain encourageant, mais insuffisant pour remplacer le modèle officiel.

---

## PHASE VIII — SOUMISSION ✅

- Fichier généré : `outputs/submission.csv`.
- Modèle de soumission : **LightGBM F, 30 features structurées, n_estimators=2000**.
- Score public Challenge Data : **MAPE = 24.25**.
- Rang public : **8 / 17**.
- Benchmark officiel : **68.14**, battu par un facteur d'environ **2.8×**.

---

## COMMANDES DE REPRODUCTION

Depuis la racine du repo :

```bash
python -m pip install -r requirements.txt
cd notebooks
jupyter nbconvert --to notebook --execute main.ipynb --inplace
```

Nettoyer les outputs avant un run complet :

```bash
find outputs -mindepth 1 -delete
```

---

## OUTPUTS ATTENDUS

```text
outputs/submission.csv
outputs/diagnostic_lightgbm_f.png
outputs/mape_by_quartile_lightgbm_f.png
outputs/examples_predictions_lightgbm_f.png
outputs/robustness_holdout_apples_to_apples.csv
outputs/feature_family_ablation_holdout.csv
outputs/regime_decile_f_vs_baseline4.csv
outputs/pseudo_time_blocks_f_vs_baseline4.csv
outputs/candidate_features_cv.csv
outputs/regularized_lgbm_cv.csv
```

---

## RÈGLES STRICTES

### Validation

- ❌ Jamais de fit du scaler/PCA/statistique stock sur validation ou test.
- ❌ Jamais d'early stopping sur le holdout final.
- ❌ Pas de sélection du champion sur le diagnostic post-holdout.
- ✅ Holdout 15% réservé à l'évaluation finale.
- ✅ Bootstrap groupé par date.

### Reproductibilité

- ✅ `random_state` fixé.
- ✅ Outputs régénérés par le notebook.
- ✅ Fichier de soumission généré explicitement.
- ✅ Résultats négatifs conservés.

### Honnêteté méthodologique

- ✅ Les pistes abandonnées sont documentées.
- ✅ Les limites risk-management sont explicites.
- ✅ Le modèle officiel reste stable malgré les tests exploratoires.

---

## SOURCES

- Giuseppe Paleologo, *The Elements of Quantitative Investing*, Wiley 2024.
- F. Corsi, “A Simple Approximate Long-Memory Model of Realized Volatility,” *Journal of Financial Econometrics*, 2009.
- A. C. Harvey and N. Shephard, “Estimation of an Asymmetric Stochastic Volatility Model for Asset Returns,” 1996.
- A. J. Patton and K. Sheppard, “Evaluating Volatility and Correlation Forecasts,” 2009.
- V. Marchenko and L. Pastur, “Distribution of Eigenvalues for Some Sets of Random Matrices,” 1967.
- M. López de Prado, *Advances in Financial Machine Learning*, Wiley 2018.

---

**Dépôt GitHub** : `adamkerouredan/CFM_Vol_FCT`  
**Auteur** : Adam Kerouredan — M2 Quantitative Finance, 2025–2026.
