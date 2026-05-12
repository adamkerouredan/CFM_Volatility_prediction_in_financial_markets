"""
model.py
--------
Responsabilité unique : entraîner et appliquer des modèles de
prédiction de volatilité.

Implémente :
- RidgeModel avec choix de scaler (Standard/Robust) et winsorisation
  optionnelle.
- RidgePCAModel : pipeline avec PCA pour réduction de dimensionnalité.
- Factory functions pour le Validator.

Référence : Paleologo (2024), Chap. 3 -- Linear Models of Returns
            et Chap. 7 -- Statistical Factor Models.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from typing import Callable, Optional


class FeatureWinsorizer:
    """
    Winsorise chaque feature à un percentile donné, fittée sur le train.

    Parameters
    ----------
    upper_percentile : float
        Percentile supérieur (ex: 99.5).
    lower_percentile : float
        Percentile inférieur (ex: 0.5).
    """

    def __init__(
        self,
        upper_percentile: float = 99.5,
        lower_percentile: float = 0.5,
    ) -> None:
        self.upper_percentile = upper_percentile
        self.lower_percentile = lower_percentile
        self.upper_bounds_: Optional[np.ndarray] = None
        self.lower_bounds_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "FeatureWinsorizer":
        """Calcule les bornes par feature sur le train."""
        self.lower_bounds_ = np.percentile(X, self.lower_percentile, axis=0)
        self.upper_bounds_ = np.percentile(X, self.upper_percentile, axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Applique les bornes apprises sur le train."""
        if self.upper_bounds_ is None:
            raise RuntimeError("FeatureWinsorizer non fitté.")
        return np.clip(X, self.lower_bounds_, self.upper_bounds_)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class RidgeModel:
    """
    Pipeline : winsorisation optionnelle -> scaler -> Ridge.

    Parameters
    ----------
    alpha : float
        Force de régularisation L2.
    scaler_type : str
        "standard" ou "robust".
    winsorize : bool
        Si True, applique une winsorisation aux features.
    winsorize_upper : float
        Percentile supérieur de winsorisation (défaut 99.5).
    """

    def __init__(
        self,
        alpha: float = 1.0,
        scaler_type: str = "standard",
        winsorize: bool = False,
        winsorize_upper: float = 99.5,
    ) -> None:
        self.alpha            = alpha
        self.scaler_type      = scaler_type
        self.winsorize        = winsorize
        self.winsorize_upper  = winsorize_upper

        self.winsorizer = (
            FeatureWinsorizer(
                upper_percentile = winsorize_upper,
                lower_percentile = 100 - winsorize_upper,
            ) if winsorize else None
        )
        self.scaler = (
            RobustScaler() if scaler_type == "robust"
            else StandardScaler()
        )
        self.ridge = Ridge(alpha=alpha, fit_intercept=True)

    def fit(self, X_train: np.ndarray, y_log_train: np.ndarray) -> "RidgeModel":
        """Fit pipeline complet — uniquement sur le train."""
        X_processed = X_train
        if self.winsorize:
            X_processed = self.winsorizer.fit_transform(X_processed)
        X_scaled = self.scaler.fit_transform(X_processed)
        self.ridge.fit(X_scaled, y_log_train)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Prédiction en espace log."""
        X_processed = X
        if self.winsorize:
            X_processed = self.winsorizer.transform(X_processed)
        X_scaled = self.scaler.transform(X_processed)
        return self.ridge.predict(X_scaled)


class RidgePCAModel:
    """
    Pipeline : winsorisation -> StandardScaler -> PCA -> Ridge.

    Réduit la dimensionnalité des features via PCA avant la régression
    Ridge. Le nombre de composantes est déterminé via le critère
    de Marchenko-Pastur.

    Parameters
    ----------
    alpha : float
        Force de régularisation L2 du Ridge.
    n_components : int
        Nombre de composantes principales (défaut 3, issu de
        Marchenko-Pastur sur notre dataset).
    winsorize : bool
        Si True, applique une winsorisation aux features avant scaling.
    winsorize_upper : float
        Percentile supérieur de winsorisation.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        n_components: int = 3,
        winsorize: bool = True,
        winsorize_upper: float = 99.5,
    ) -> None:
        self.alpha            = alpha
        self.n_components     = n_components
        self.winsorize        = winsorize
        self.winsorize_upper  = winsorize_upper

        self.winsorizer = (
            FeatureWinsorizer(
                upper_percentile = winsorize_upper,
                lower_percentile = 100 - winsorize_upper,
            ) if winsorize else None
        )
        self.scaler  = StandardScaler()
        self.pca     = PCA(n_components=n_components)
        self.ridge   = Ridge(alpha=alpha, fit_intercept=True)

    def fit(self, X_train: np.ndarray, y_log_train: np.ndarray) -> "RidgePCAModel":
        """Fit pipeline complet — uniquement sur le train."""
        X_processed = X_train
        if self.winsorize:
            X_processed = self.winsorizer.fit_transform(X_processed)
        X_scaled = self.scaler.fit_transform(X_processed)
        X_pca    = self.pca.fit_transform(X_scaled)
        self.ridge.fit(X_pca, y_log_train)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Prédiction en espace log."""
        X_processed = X
        if self.winsorize:
            X_processed = self.winsorizer.transform(X_processed)
        X_scaled = self.scaler.transform(X_processed)
        X_pca    = self.pca.transform(X_scaled)
        return self.ridge.predict(X_pca)

    def get_explained_variance_ratio(self) -> np.ndarray:
        """Retourne le ratio de variance expliquée par chaque composante."""
        return self.pca.explained_variance_ratio_

    def get_loadings(self) -> np.ndarray:
        """Retourne la matrice de loadings (composantes principales)."""
        return self.pca.components_


class LightGBMModel:
    """
    Pipeline LightGBM avec early stopping interne pour éviter
    le leakage et l'overfitting.

    LightGBM gère nativement les NaN et est invariant par
    transformation monotone — pas de winsorisation ni de scaling
    nécessaire.

    Parameters
    ----------
    n_estimators : int
        Nombre maximum d'arbres (early stopping peut couper avant).
    max_depth : int
        Profondeur maximale par arbre.
    num_leaves : int
        Nombre maximum de feuilles par arbre.
    learning_rate : float
        Pas d'apprentissage.
    min_child_samples : int
        Nombre minimum d'observations par feuille (régularisation).
    reg_alpha : float
        Pénalisation L1 sur les feuilles.
    reg_lambda : float
        Pénalisation L2 sur les feuilles.
    early_stopping_rounds : int
        Nombre d'itérations sans amélioration avant arrêt.
    internal_val_size : float
        Fraction du train utilisée comme validation interne pour
        l'early stopping (jamais le fold de test externe).
    random_state : int
        Seed pour la reproductibilité.
    categorical_features : list, optional
        Liste des indices des colonnes catégorielles.
    """

    def __init__(
        self,
        n_estimators: int             = 500,
        max_depth: int                = 5,
        num_leaves: int               = 31,
        learning_rate: float          = 0.05,
        min_child_samples: int        = 50,
        reg_alpha: float              = 0.0,
        reg_lambda: float             = 1.0,
        early_stopping_rounds: int    = 20,
        internal_val_size: float      = 0.15,
        random_state: int             = 42,
        categorical_features: list    = None,
    ) -> None:
        self.n_estimators           = n_estimators
        self.max_depth              = max_depth
        self.num_leaves             = num_leaves
        self.learning_rate          = learning_rate
        self.min_child_samples      = min_child_samples
        self.reg_alpha              = reg_alpha
        self.reg_lambda             = reg_lambda
        self.early_stopping_rounds  = early_stopping_rounds
        self.internal_val_size      = internal_val_size
        self.random_state           = random_state
        self.categorical_features   = categorical_features

        self.model_                 = None
        self.best_iteration_        = None

    def fit(
        self,
        X_train: np.ndarray,
        y_log_train: np.ndarray,
    ) -> "LightGBMModel":
        """
        Entraînement avec early stopping sur un sous-fold interne.
        """
        import lightgbm as lgb
        from sklearn.model_selection import train_test_split

        X_internal_train, X_internal_val, y_internal_train, y_internal_val = (
            train_test_split(
                X_train, y_log_train,
                test_size    = self.internal_val_size,
                random_state = self.random_state,
            )
        )

        train_set = lgb.Dataset(
            X_internal_train,
            label                 = y_internal_train,
            categorical_feature   = self.categorical_features,
        )
        val_set = lgb.Dataset(
            X_internal_val,
            label                 = y_internal_val,
            reference             = train_set,
            categorical_feature   = self.categorical_features,
        )

        params = {
            "objective":          "regression",
            "metric":             "rmse",
            "max_depth":          self.max_depth,
            "num_leaves":         self.num_leaves,
            "learning_rate":      self.learning_rate,
            "min_child_samples":  self.min_child_samples,
            "reg_alpha":          self.reg_alpha,
            "reg_lambda":         self.reg_lambda,
            "verbose":            -1,
            "seed":               self.random_state,
        }

        self.model_ = lgb.train(
            params,
            train_set,
            num_boost_round   = self.n_estimators,
            valid_sets        = [val_set],
            valid_names       = ["val_internal"],
            callbacks         = [
                lgb.early_stopping(self.early_stopping_rounds, verbose=False),
                lgb.log_evaluation(0),
            ],
        )

        self.best_iteration_ = self.model_.best_iteration
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Prédiction en espace log avec le meilleur nombre d'arbres."""
        if self.model_ is None:
            raise RuntimeError("Le modèle n'est pas entraîné.")
        return self.model_.predict(X, num_iteration=self.best_iteration_)

    def get_feature_importance(
        self,
        feature_names: list,
        importance_type: str = "gain",
    ):
        """
        Retourne l'importance des features triée par ordre décroissant.

        Parameters
        ----------
        feature_names : list
            Noms des features dans l'ordre des colonnes de X.
        importance_type : str
            "gain" (par défaut) ou "split".
        """
        import pandas as pd
        if self.model_ is None:
            raise RuntimeError("Le modèle n'est pas entraîné.")

        importances = self.model_.feature_importance(
            importance_type=importance_type
        )
        return (
            pd.DataFrame({
                "feature":    feature_names,
                "importance": importances,
            })
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
    
class HARLightGBMStackingModel:
    """
    Residual stacking : HAR-RV linéaire + LightGBM sur les résidus.

    Étape 1 : HAR-RV (Ridge sur 3 horizons log) capture la
              composante autorégressive linéaire.
    Étape 2 : LightGBM sur 10 features engineered capture les
              non-linéarités résiduelles.

    Input X attendu : matrice (n, 13) = 3 features HAR + 10 features.
    L'ordre des colonnes est fixé par la fonction d'assemblage.

    Parameters
    ----------
    har_alpha : float
        Régularisation Ridge du modèle HAR.
    lgbm_params : dict
        Hyperparamètres LightGBM pour la couche résiduelle.
    """

    def __init__(
        self,
        har_alpha: float    = 0.001,
        lgbm_params: dict   = None,
        n_har_features: int = 3,
    ) -> None:
        self.har_alpha       = har_alpha
        self.n_har_features  = n_har_features
        self.lgbm_params     = lgbm_params or {
            "n_estimators":           1000,
            "max_depth":              5,
            "num_leaves":             31,
            "learning_rate":          0.03,
            "min_child_samples":      50,
            "reg_alpha":              0.0,
            "reg_lambda":             1.0,
            "early_stopping_rounds":  30,
            "internal_val_size":      0.15,
            "random_state":           42,
        }
        self.har_model_  = None
        self.lgbm_model_ = None

    def fit(
        self,
        X_combined: np.ndarray,
        y_log_train: np.ndarray,
    ) -> "HARLightGBMStackingModel":
        """
        Entraîne la stack sur le train uniquement.

        X_combined : (n, 13) = [features HAR | features engineered]
        """
        # Split des deux jeux de features
        X_har        = X_combined[:, :self.n_har_features]
        X_lgbm_feats = X_combined[:, self.n_har_features:]

        # ÉTAPE 1 : HAR-RV
        self.har_model_ = RidgeModel(
            alpha           = self.har_alpha,
            scaler_type     = "standard",
            winsorize       = True,
            winsorize_upper = 99.5,
        )
        self.har_model_.fit(X_har, y_log_train)

        # ÉTAPE 2 : résidus dans l'espace log
        y_har_pred = self.har_model_.predict(X_har)
        residuals  = y_log_train - y_har_pred

        # ÉTAPE 3 : LightGBM sur les résidus
        self.lgbm_model_ = LightGBMModel(**self.lgbm_params)
        self.lgbm_model_.fit(X_lgbm_feats, residuals)

        return self

    def predict(self, X_combined: np.ndarray) -> np.ndarray:
        """Prédiction finale = HAR + LightGBM(résidus)."""
        if self.har_model_ is None or self.lgbm_model_ is None:
            raise RuntimeError("Le modèle n'est pas entraîné.")

        X_har        = X_combined[:, :self.n_har_features]
        X_lgbm_feats = X_combined[:, self.n_har_features:]

        y_har_pred       = self.har_model_.predict(X_har)
        residuals_pred   = self.lgbm_model_.predict(X_lgbm_feats)

        return y_har_pred + residuals_pred



def make_lightgbm_train_fn(
    n_estimators: int             = 500,
    max_depth: int                = 5,
    num_leaves: int               = 31,
    learning_rate: float          = 0.05,
    min_child_samples: int        = 50,
    reg_alpha: float              = 0.0,
    reg_lambda: float             = 1.0,
    early_stopping_rounds: int    = 20,
    internal_val_size: float      = 0.15,
    random_state: int             = 42,
    categorical_features: list    = None,   # ← AJOUT
) -> Callable:
    """Factory : crée une fonction d'entraînement LightGBM."""
    def train_fn(X_train: np.ndarray, y_log_train: np.ndarray) -> LightGBMModel:
        model = LightGBMModel(
            n_estimators          = n_estimators,
            max_depth             = max_depth,
            num_leaves            = num_leaves,
            learning_rate         = learning_rate,
            min_child_samples     = min_child_samples,
            reg_alpha             = reg_alpha,
            reg_lambda            = reg_lambda,
            early_stopping_rounds = early_stopping_rounds,
            internal_val_size     = internal_val_size,
            random_state          = random_state,
            categorical_features  = categorical_features,   # ← AJOUT
        )
        return model.fit(X_train, y_log_train)
    return train_fn

# ----------------------------------------------------------------------
# Factory functions pour le Validator
# ----------------------------------------------------------------------

def make_ridge_train_fn(
    alpha: float,
    scaler_type: str = "standard",
    winsorize: bool = False,
    winsorize_upper: float = 99.5,
) -> Callable:
    """Factory : crée une fonction d'entraînement Ridge."""
    def train_fn(X_train: np.ndarray, y_log_train: np.ndarray) -> RidgeModel:
        model = RidgeModel(
            alpha           = alpha,
            scaler_type     = scaler_type,
            winsorize       = winsorize,
            winsorize_upper = winsorize_upper,
        )
        return model.fit(X_train, y_log_train)
    return train_fn


def make_ridge_pca_train_fn(
    alpha: float,
    n_components: int = 3,
    winsorize: bool = True,
    winsorize_upper: float = 99.5,
) -> Callable:
    """Factory : crée une fonction d'entraînement Ridge + PCA."""
    def train_fn(X_train: np.ndarray, y_log_train: np.ndarray) -> RidgePCAModel:
        model = RidgePCAModel(
            alpha           = alpha,
            n_components    = n_components,
            winsorize       = winsorize,
            winsorize_upper = winsorize_upper,
        )
        return model.fit(X_train, y_log_train)
    return train_fn

def make_har_lightgbm_train_fn(
    har_alpha: float = 0.001,
    lgbm_params: dict = None,
) -> Callable:
    """Factory : crée une fonction d'entraînement HAR + LightGBM stacking."""
    def train_fn(X_combined: np.ndarray, y_log_train: np.ndarray):
        model = HARLightGBMStackingModel(
            har_alpha   = har_alpha,
            lgbm_params = lgbm_params,
        )
        return model.fit(X_combined, y_log_train)
    return train_fn


def predict_fn(model, X_val: np.ndarray) -> np.ndarray:
    """Fonction de prédiction commune (espace log)."""
    return model.predict(X_val)