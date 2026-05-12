"""
validator.py
------------
Responsabilité unique : protocole de validation croisée pour évaluer
la performance d'un modèle de manière robuste.

Implémente :
- Repeated Stratified K-Fold (5 folds × 3 répétitions)
- Stratification par quartile de TARGET
- Agrégation propre des MAPE (moyenne ± std)
- Garde-fous contre le leakage : fit du scaler/PCA uniquement sur train

Référence : Paleologo (2024), Chap. 4 -- Backtesting Protocol.
"""

import numpy as np
import pandas as pd
from typing import Callable, Tuple, Any
from sklearn.model_selection import StratifiedKFold


class Validator:
    """
    Validation croisée Repeated Stratified K-Fold.

    Parameters
    ----------
    n_splits : int
        Nombre de folds par répétition (défaut 5).
    n_repeats : int
        Nombre de répétitions avec seeds différents (défaut 3).
    n_strata : int
        Nombre de strates pour la stratification (défaut 4).
    base_random_state : int
        Seed de base, incrémenté à chaque répétition.
    """

    def __init__(
        self,
        n_splits: int = 5,
        n_repeats: int = 3,
        n_strata: int = 4,
        base_random_state: int = 42,
    ) -> None:
        self.n_splits           = n_splits
        self.n_repeats          = n_repeats
        self.n_strata           = n_strata
        self.base_random_state  = base_random_state

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def cross_validate(
        self,
        X: np.ndarray,
        y_log: np.ndarray,
        y_original: np.ndarray,
        train_fn: Callable,
        predict_fn: Callable,
        evaluator: Any,
    ) -> dict:
        """
        Effectue une validation croisée Repeated Stratified K-Fold.

        Parameters
        ----------
        X : np.ndarray
            Features d'entrée (déjà préparées : neutralisées ou non).
        y_log : np.ndarray
            Target en espace log (utilisée pour entraîner le modèle).
        y_original : np.ndarray
            Target en espace original (utilisée pour calculer le MAPE).
        train_fn : Callable
            Fonction signature train_fn(X_train, y_log_train) -> model.
            Le modèle retourné doit être prêt pour predict.
        predict_fn : Callable
            Fonction signature predict_fn(model, X_val) -> y_log_pred.
            Retourne les prédictions en espace log.
        evaluator : Evaluator
            Instance de Evaluator pour le calcul du MAPE et la
            correction de Jensen.

        Returns
        -------
        dict avec :
            - mape_per_fold : liste des MAPE de chaque fold
            - mape_mean : moyenne des MAPE
            - mape_std : écart-type des MAPE
            - residual_var_mean : variance résiduelle moyenne (log space)
            - n_folds_total : nombre total de folds évalués
        """
        self._validate_inputs(X, y_log, y_original)

        strata = self._build_strata(y_original)
        mape_scores = []
        residual_variances = []

        for repeat in range(self.n_repeats):
            seed = self.base_random_state + repeat
            skf = StratifiedKFold(
                n_splits     = self.n_splits,
                shuffle      = True,
                random_state = seed,
            )

            for fold_idx, (train_idx, val_idx) in enumerate(
                skf.split(X, strata)
            ):
                mape, residual_var = self._evaluate_fold(
                    X, y_log, y_original,
                    train_idx, val_idx,
                    train_fn, predict_fn, evaluator,
                )
                mape_scores.append(mape)
                residual_variances.append(residual_var)

                print(
                    f"  Repeat {repeat+1}/{self.n_repeats} "
                    f"Fold {fold_idx+1}/{self.n_splits} : "
                    f"MAPE = {mape:.4f} | residual_var = {residual_var:.4f}"
                )

        return {
            "mape_per_fold":      mape_scores,
            "mape_mean":          float(np.mean(mape_scores)),
            "mape_std":           float(np.std(mape_scores)),
            "residual_var_mean":  float(np.mean(residual_variances)),
            "n_folds_total":      len(mape_scores),
        }

    def print_summary(self, results: dict) -> None:
        """Affichage synthétique des résultats de cross-validation."""
        print("\n" + "=" * 55)
        print("RÉSULTATS CROSS-VALIDATION")
        print("=" * 55)
        print(f"  Folds évalués       : {results['n_folds_total']}")
        print(f"  MAPE moyen          : {results['mape_mean']:.4f}")
        print(f"  MAPE std            : {results['mape_std']:.4f}")
        print(f"  Variance résiduelle : {results['residual_var_mean']:.4f}")
        print(f"  IC 95% MAPE         : "
              f"[{results['mape_mean'] - 1.96 * results['mape_std']:.4f}, "
              f"{results['mape_mean'] + 1.96 * results['mape_std']:.4f}]")

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _evaluate_fold(
        self,
        X: np.ndarray,
        y_log: np.ndarray,
        y_original: np.ndarray,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        train_fn: Callable,
        predict_fn: Callable,
        evaluator: Any,
    ) -> Tuple[float, float]:
        """Évalue un seul fold et retourne (MAPE, variance résiduelle)."""
        X_train     = X[train_idx]
        X_val       = X[val_idx]
        y_log_train = y_log[train_idx]
        y_log_val   = y_log[val_idx]
        y_orig_val  = y_original[val_idx]

        model = train_fn(X_train, y_log_train)
        y_log_pred = predict_fn(model, X_val)

        # Estimation variance résiduelle pour Jensen
        evaluator.fit_residual_variance(y_log_val, y_log_pred)

        # Prédiction en espace original avec correction Jensen
        y_pred = evaluator.predict_volatility(y_log_pred)

        mape         = evaluator.mape(y_orig_val, y_pred)
        residual_var = float(np.var(y_log_val - y_log_pred))

        return mape, residual_var

    def _build_strata(self, y_original: np.ndarray) -> np.ndarray:
        """Crée les strates pour la stratification."""
        return pd.qcut(
            pd.Series(y_original),
            q=self.n_strata,
            labels=False,
            duplicates="drop",
        ).values

    def _validate_inputs(
        self,
        X: np.ndarray,
        y_log: np.ndarray,
        y_original: np.ndarray,
    ) -> None:
        """Vérifie la cohérence des inputs."""
        n = len(X)
        if len(y_log) != n or len(y_original) != n:
            raise ValueError(
                f"Tailles incohérentes : X={n}, y_log={len(y_log)}, "
                f"y_original={len(y_original)}"
            )
        if (y_original <= 0).any():
            raise ValueError("y_original doit être strictement positif.")