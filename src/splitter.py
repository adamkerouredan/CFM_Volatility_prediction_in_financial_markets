"""
splitter.py
-----------
Responsabilité unique : créer les splits train / holdout de manière
reproductible et stratifiée.

Le holdout est intouchable jusqu'à la validation finale.
Aucun modèle, aucun hyperparamètre, aucune feature ne peut être
optimisé sur le holdout.

Référence : Paleologo (2024), Chapitre 4 -- Backtesting Protocol.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from typing import Tuple


class Splitter:
    """
    Crée un split stratifié train / holdout par quartile de TARGET.

    Parameters
    ----------
    holdout_size : float
        Proportion du dataset réservée au holdout (par défaut 0.15).
    n_strata : int
        Nombre de strates pour la stratification (par défaut 4 = quartiles).
    random_state : int
        Seed pour la reproductibilité.
    """

    def __init__(
        self,
        holdout_size: float = 0.15,
        n_strata: int = 4,
        random_state: int = 42,
    ) -> None:
        self.holdout_size  = holdout_size
        self.n_strata      = n_strata
        self.random_state  = random_state

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def split(
        self,
        features: pd.DataFrame,
        target: pd.Series,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Crée les indices train / holdout stratifiés.

        Parameters
        ----------
        features : pd.DataFrame
            Features (doit contenir la colonne ID).
        target : pd.Series
            Target d'entraînement (TARGET originale, pas log).

        Returns
        -------
        train_idx, holdout_idx : np.ndarray
            Indices entiers vers les lignes de train et holdout.
        """
        if len(features) != len(target):
            raise ValueError(
                f"Features ({len(features)}) et target ({len(target)}) "
                "n'ont pas la même longueur."
            )

        strata = self._build_strata(target)

        splitter = StratifiedShuffleSplit(
            n_splits     = 1,
            test_size    = self.holdout_size,
            random_state = self.random_state,
        )

        train_idx, holdout_idx = next(
            splitter.split(np.zeros(len(features)), strata)
        )
        return train_idx, holdout_idx

    def diagnose(
        self,
        target: pd.Series,
        train_idx: np.ndarray,
        holdout_idx: np.ndarray,
    ) -> None:
        """Affiche un diagnostic du split — distribution comparée."""
        target_arr = target.values

        print("=" * 55)
        print("DIAGNOSTIC DU SPLIT")
        print("=" * 55)
        print(f"  Train   : {len(train_idx):>8,} ({len(train_idx)/len(target):.1%})")
        print(f"  Holdout : {len(holdout_idx):>8,} ({len(holdout_idx)/len(target):.1%})")

        train_target   = target_arr[train_idx]
        holdout_target = target_arr[holdout_idx]

        print("\n  Distribution TARGET (train vs holdout) :")
        print(f"  {'Statistique':<15} {'Train':>12} {'Holdout':>12}")
        print(f"  {'-'*15} {'-'*12} {'-'*12}")
        for stat_name, stat_fn in [
            ("mean",     np.mean),
            ("median",   np.median),
            ("std",      np.std),
            ("min",      np.min),
            ("max",      np.max),
            ("p99",      lambda x: np.percentile(x, 99)),
        ]:
            print(
                f"  {stat_name:<15} "
                f"{stat_fn(train_target):>12.4f} "
                f"{stat_fn(holdout_target):>12.4f}"
            )

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _build_strata(self, target: pd.Series) -> np.ndarray:
        """
        Crée les strates par quantile de la TARGET.
        Stratification = chaque strate est équilibrée entre train et holdout.
        """
        strata = pd.qcut(
            target,
            q=self.n_strata,
            labels=False,
            duplicates="drop",
        )
        return strata.values