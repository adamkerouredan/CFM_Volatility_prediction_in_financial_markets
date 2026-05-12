"""
feature_engineer.py
-------------------
Responsabilité unique : imputer les NaN et construire les features
économiquement motivées à partir des données brutes.

Étape 4 du process de Paleologo (2024) : Loadings Generation.
"""

import numpy as np
import pandas as pd
from typing import Tuple


class FeatureEngineer:
    """
    Impute les valeurs manquantes et génère les features
    pour le challenge CFM de prédiction de volatilité.

    Parameters
    ----------
    x_train : pd.DataFrame
        Données brutes d'entraînement.
    x_test : pd.DataFrame
        Données brutes de test.
    n_recent_bars : int
        Nombre de barres récentes pour les features de récence.
    n_early_bars : int
        Nombre de barres initiales pour les features de tendance.
    """

    VOLATILITY_PREFIX = "volatility"
    RETURN_PREFIX     = "return"
    META_COLUMNS      = ["ID", "date", "product_id"]

    def __init__(
        self,
        x_train: pd.DataFrame,
        x_test: pd.DataFrame,
        n_recent_bars: int = 6,
        n_early_bars: int  = 6,
    ) -> None:
        self.x_train       = x_train.copy()
        self.x_test        = x_test.copy()
        self.n_recent_bars = n_recent_bars
        self.n_early_bars  = n_early_bars

        self.volatility_cols = self._get_columns(self.VOLATILITY_PREFIX)
        self.return_cols     = self._get_columns(self.RETURN_PREFIX)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Pipeline complet : imputation puis construction des features.

        Returns
        -------
        features_train, features_test : DataFrames de features propres
        """
        x_train_imputed = self._impute(self.x_train)
        x_test_imputed  = self._impute(self.x_test)

        features_train = self._build_features(x_train_imputed)
        features_test  = self._build_features(x_test_imputed)

        return features_train, features_test

    # ------------------------------------------------------------------
    # Private : imputation
    # ------------------------------------------------------------------

    def _impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute les NaN sur les colonnes de volatilité par interpolation
        linéaire intraday + ffill + bfill.

        Impute les NaN sur les colonnes de retour par 0 (absence de
        mouvement — valeur neutre économiquement justifiée).

        Référence : Paleologo (2024) -- simplicité défendable préférée
        à la sophistication à faible valeur ajoutée.
        """
        df = df.copy()

        # Imputation volatilité — interpolation intraday
        df[self.volatility_cols] = (
            df[self.volatility_cols]
            .interpolate(method="linear", axis=1)
            .ffill(axis=1)
            .bfill(axis=1)
        )

        # Imputation retours — fill par 0 (pas de mouvement)
        df[self.return_cols] = (
            df[self.return_cols]
            .fillna(0)
        )

        return df

    # ------------------------------------------------------------------
    # Private : construction des features
    # ------------------------------------------------------------------

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Construit toutes les features et retourne un DataFrame propre.
        Chaque famille de features a une hypothèse économique explicite.
        """
        features = pd.DataFrame(index=df.index)
        features["ID"] = df["ID"]

        vol  = df[self.volatility_cols]
        ret  = df[self.return_cols]

        recent_vol = vol[self.volatility_cols[-self.n_recent_bars:]]
        early_vol  = vol[self.volatility_cols[:self.n_early_bars]]
        mid_vol    = vol[self.volatility_cols[
            self.n_early_bars : len(self.volatility_cols) - self.n_recent_bars
        ]]

        # --- Famille 1 : Niveau global ---
        # Hypothèse : persistance de volatilité (r=0.857 observé en EDA)
        features["vol_mean"]   = vol.mean(axis=1)
        features["vol_median"] = vol.median(axis=1)

        # --- Famille 2 : Asymétrie intraday ---
        # Hypothèse : l'écart moyenne/médiane mesure la présence de pics
        # isolés. Un jour avec un pic violent à l'ouverture a un profil
        # de risque différent d'un jour uniformément volatile.
        features["vol_mean_minus_median"] = (
            features["vol_mean"] - features["vol_median"]
        )

        # --- Famille 3 : Niveau récent ---
        # Hypothèse : les barres proches de 14h sont les plus prédictives
        # car elles reflètent le régime de volatilité actuel de la séance.
        features["vol_mean_recent"] = recent_vol.mean(axis=1)
        features["vol_last_bar"]    = vol[self.volatility_cols[-1]]

        # --- Famille 4 : Tendance intraday ---
        # Hypothèse : si la vol remonte en fin de matinée (vol_trend > 0),
        # l'après-midi sera probablement plus volatile.
        features["vol_trend"] = (
            recent_vol.mean(axis=1) - early_vol.mean(axis=1)
        )

        # Pente d'une régression linéaire sur les 54 barres
        # Capture la direction globale du profil intraday
        features["vol_linear_slope"] = self._compute_linear_slope(vol)

        # --- Famille 5 : Dispersion ---
        # Hypothèse : une vol erratique (std élevé) est plus difficile
        # à prédire et signale un régime de marché instable.
        features["vol_std"]   = vol.std(axis=1)
        features["vol_max"]   = vol.max(axis=1)
        features["vol_min"]   = vol.min(axis=1)
        features["vol_range"] = features["vol_max"] - features["vol_min"]

        # --- Famille 6 : Ratio récence / global ---
        # Hypothèse : si la vol récente est bien au-dessus de la moyenne,
        # le signal s'accélère vers 14h.
        features["vol_recent_over_mean"] = (
            features["vol_mean_recent"] / (features["vol_mean"] + 1e-8)
        )

        # --- Famille 7 : Information directionnelle ---
        # Hypothèse : la proportion de retours positifs et les runs
        # directionnels contiennent une information sur le momentum
        # intraday, potentiellement corrélée au niveau de volatilité.
        features["return_n_positive"]    = (ret == 1).sum(axis=1)
        features["return_n_negative"]    = (ret == -1).sum(axis=1)
        features["return_direction_bias"] = (
            features["return_n_positive"] - features["return_n_negative"]
        ) / ret.shape[1]
        features["return_last_bar"]      = ret[self.return_cols[-1]]

        # --- Suppression des features redondantes ou sans signal ---
        # vol_median     : r=0.969 avec vol_mean (Pearson)
        # vol_trend      : r=0.918 avec vol_linear_slope (Pearson)
        # return_direction_bias : IC=-0.048 → sans signal
        # return_last_bar       : IC=+0.002 → sans signal
        # vol_range      : r=0.999 avec vol_max (Spearman) → détecté post-analyse
        # vol_max        : r=0.956 avec vol_std (Spearman) → redondant avec vol_std
        features = features.drop(columns=[
            "vol_median",
            "vol_trend",
            "return_direction_bias",
            "return_last_bar",
            "vol_range",
            "vol_max",
        ])

        return features.reset_index(drop=True)


    def _compute_linear_slope(self, vol: pd.DataFrame) -> pd.Series:
        """
        Calcule la pente d'une régression linéaire OLS sur les 54 barres
        pour chaque ligne. Capture la tendance directionnelle du profil
        intraday de volatilité.

        Returns
        -------
        pd.Series : pente pour chaque observation
        """
        n     = vol.shape[1]
        x     = np.arange(n, dtype=float)
        x_c   = x - x.mean()
        denom = (x_c ** 2).sum()

        values = vol.values
        y_c    = values - values.mean(axis=1, keepdims=True)
        slopes = (y_c * x_c).sum(axis=1) / denom

        return pd.Series(slopes, index=vol.index)

    # ------------------------------------------------------------------
    # Private : utilitaire
    # ------------------------------------------------------------------

    def _get_columns(self, prefix: str) -> list[str]:
        """Retourne les colonnes dont le nom commence par le préfixe."""
        return [
            col for col in self.x_train.columns
            if col.startswith(prefix)
        ]


class FeatureTransformer:
    """
    Génère des transformations cross-sectionnelles et historiques
    des features brutes selon Paleologo (2024), Chap. 6 — Loadings
    Generation.

    Pour chaque feature de base, construit :
    - Z-score cross-sectionnel par date (median + MAD, robuste outliers)
    - Z-score historique par stock (mean/std fittés sur train)

    Discipline anti-leakage :
    - Les Z-scores par date n'utilisent JAMAIS la target → autorisés
      sur tout l'échantillon (train + test).
    - Les Z-scores par stock utilisent uniquement les statistiques
      calculées sur le train (fit_stock_stats).
    """

    def __init__(self, feature_cols: list) -> None:
        self.feature_cols   = feature_cols
        self.stock_stats_   = {}   # {col: DataFrame avec mean/std par stock}
        self._is_fitted     = False

    def fit_stock_stats(
        self,
        features_train: pd.DataFrame,
        meta_train: pd.DataFrame,
    ) -> "FeatureTransformer":
        """
        Calcule mean et std de chaque feature par stock,
        uniquement sur le train.
        """
        df = features_train[self.feature_cols].copy()
        df["product_id"] = meta_train["product_id"].values

        for col in self.feature_cols:
            stats = df.groupby("product_id")[col].agg(["mean", "std"])
            stats["std"] = stats["std"].replace(0, 1e-8)  # éviter div par 0
            self.stock_stats_[col] = stats

        self._is_fitted = True
        return self

    def transform_zscore_date(
        self,
        features: pd.DataFrame,
        meta: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Z-score cross-sectionnel par date.
        Utilise median + MAD pour robustesse aux outliers.
        Aucune fuite TARGET → utilisable sur tout l'échantillon.
        """
        df = features[self.feature_cols].copy()
        df["date"] = meta["date"].values

        result = pd.DataFrame(index=features.index)
        for col in self.feature_cols:
            grouped = df.groupby("date")[col]
            median  = grouped.transform("median")
            mad     = grouped.transform(
                lambda x: np.median(np.abs(x - x.median()))
            )
            # 1.4826 = facteur pour rendre MAD comparable à std gaussien
            result[f"{col}_zd"] = (df[col] - median) / (1.4826 * mad + 1e-8)

        return result

    def transform_zscore_stock(
        self,
        features: pd.DataFrame,
        meta: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Z-score historique par stock.
        Utilise les statistiques fittées sur le train uniquement.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "FeatureTransformer doit être fitté avec "
                "fit_stock_stats() avant transform_zscore_stock()."
            )

        df = features[self.feature_cols].copy()
        df["product_id"] = meta["product_id"].values

        result = pd.DataFrame(index=features.index)
        for col in self.feature_cols:
            stats         = self.stock_stats_[col]
            mean_by_stock = df["product_id"].map(stats["mean"])
            std_by_stock  = df["product_id"].map(stats["std"])
            # Fallback : stocks jamais vus en train (rare mais possible)
            mean_by_stock = mean_by_stock.fillna(df[col].mean())
            std_by_stock  = std_by_stock.fillna(df[col].std()).replace(0, 1e-8)

            result[f"{col}_zs"] = (df[col] - mean_by_stock) / std_by_stock

        return result

    def build_full_matrix(
        self,
        features: pd.DataFrame,
        meta: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Construit la matrice complète : brut + Z-date + Z-stock.
        """
        f_zd = self.transform_zscore_date(features, meta)
        f_zs = self.transform_zscore_stock(features, meta)

        result = pd.concat([
            features[self.feature_cols].reset_index(drop=True),
            f_zd.reset_index(drop=True),
            f_zs.reset_index(drop=True),
        ], axis=1)

        return result