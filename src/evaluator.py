"""
evaluator.py
------------
Responsabilité unique : évaluer la qualité d'un modèle de prédiction
de volatilité.

Fonctionnalités :
- MAPE sur TARGET originale
- Negative Log-Likelihood sous hypothèse log-normale
- Correction de Jensen pour la reconversion log -> niveau
- Diagnostic des résidus

Référence : Paleologo (2024), Chap. 4 (Backtesting Protocol)
            et Chap. 8 (Information Coefficient).
"""

import numpy as np
import pandas as pd
from typing import Optional


class Evaluator:
    """
    Évalue la performance d'un modèle de prédiction de volatilité.

    Le modèle est entraîné dans l'espace log. Cette classe gère :
    - la reconversion exp() avec correction de Jensen,
    - le calcul du MAPE sur l'échelle originale,
    - le calcul de la NLL sous hypothèse log-normale,
    - le diagnostic des résidus.

    Parameters
    ----------
    apply_jensen_correction : bool
        Si True, applique la correction E[exp(X)] = exp(mu + sigma²/2).
    """

    def __init__(self, apply_jensen_correction: bool = True) -> None:
        self.apply_jensen_correction = apply_jensen_correction
        self._residual_variance: Optional[float] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_residual_variance(
        self,
        y_log_true: np.ndarray,
        y_log_pred: np.ndarray,
    ) -> None:
        """
        Estime la variance des résidus en espace log sur un fold de
        validation, pour la correction de Jensen.

        Doit être appelé avant `predict_volatility` si la correction
        est activée.
        """
        residuals = y_log_true - y_log_pred
        self._residual_variance = float(np.var(residuals))

    def predict_volatility(self, y_log_pred: np.ndarray) -> np.ndarray:
        """
        Convertit une prédiction en espace log vers l'espace original.

        Si la correction de Jensen est activée :
            sigma_hat = exp(y_log_pred + sigma²_residus / 2)

        Sinon :
            sigma_hat = exp(y_log_pred)
        """
        if self.apply_jensen_correction:
            if self._residual_variance is None:
                raise RuntimeError(
                    "Variance résiduelle non estimée. "
                    "Appeler fit_residual_variance() avant."
                )
            return np.exp(y_log_pred + self._residual_variance / 2.0)
        return np.exp(y_log_pred)

    def mape(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> float:
        """
        Mean Absolute Percentage Error sur l'échelle originale.

        MAPE = mean( |y_true - y_pred| / y_true )
        """
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)

        if (y_true <= 0).any():
            raise ValueError("Toutes les valeurs y_true doivent être > 0.")

        return float(np.mean(np.abs(y_true - y_pred) / y_true))

    def negative_log_likelihood(
        self,
        y_log_true: np.ndarray,
        y_log_pred: np.ndarray,
    ) -> float:
        """
        NLL sous hypothèse log-normale. Proper scoring rule alternative
        au MAPE, plus stable sur les petites valeurs.

        NLL = (1/n) * sum( 0.5 * log(2*pi*sigma²) + (y_true - y_pred)² / (2*sigma²) )
        """
        residuals  = y_log_true - y_log_pred
        sigma2     = float(np.var(residuals))
        n          = len(residuals)

        nll = 0.5 * np.log(2 * np.pi * sigma2) \
              + np.sum(residuals ** 2) / (2 * sigma2 * n)
        return float(nll)

    def diagnose_residuals(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        meta: Optional[pd.DataFrame] = None,
    ) -> dict:
        """
        Diagnostic complet des résidus en espace original.

        Parameters
        ----------
        y_true : array-like
            TARGET réelle.
        y_pred : array-like
            TARGET prédite (échelle originale, après reconversion).
        meta : pd.DataFrame, optional
            DataFrame contenant date et product_id pour les diagnostics
            par groupe.

        Returns
        -------
        dict avec les statistiques de diagnostic.
        """
        residuals = y_true - y_pred
        relative_errors = np.abs(residuals) / y_true

        diagnostics = {
            "mape":              float(np.mean(relative_errors)),
            "median_rel_error":  float(np.median(relative_errors)),
            "max_rel_error":     float(np.max(relative_errors)),
            "residual_mean":     float(np.mean(residuals)),
            "residual_std":      float(np.std(residuals)),
            "residual_skew":     float(pd.Series(residuals).skew()),
            "residual_kurt":     float(pd.Series(residuals).kurt()),
        }

        # Performance par quartile de TARGET
        quartiles = pd.qcut(y_true, q=4, labels=False, duplicates="drop")
        mape_by_quartile = []
        for q in range(4):
            mask = quartiles == q
            if mask.sum() > 0:
                mape_q = np.mean(relative_errors[mask])
                mape_by_quartile.append(float(mape_q))
        diagnostics["mape_by_quartile"] = mape_by_quartile

        return diagnostics

    def print_diagnostics(self, diagnostics: dict) -> None:
        """Affichage formaté des diagnostics."""
        print("=" * 50)
        print("DIAGNOSTIC DU MODÈLE")
        print("=" * 50)
        print(f"  MAPE global              : {diagnostics['mape']:.4f}")
        print(f"  Erreur relative médiane  : {diagnostics['median_rel_error']:.4f}")
        print(f"  Erreur relative max      : {diagnostics['max_rel_error']:.4f}")
        print(f"  Résidu moyen             : {diagnostics['residual_mean']:+.4f}")
        print(f"  Résidu std               : {diagnostics['residual_std']:.4f}")
        print(f"  Résidu skewness          : {diagnostics['residual_skew']:+.4f}")
        print(f"  Résidu kurtosis          : {diagnostics['residual_kurt']:+.4f}")
        print(f"\n  MAPE par quartile de TARGET :")
        for i, mape_q in enumerate(diagnostics['mape_by_quartile']):
            print(f"    Q{i+1} (quartile {i+1}/4) : {mape_q:.4f}")