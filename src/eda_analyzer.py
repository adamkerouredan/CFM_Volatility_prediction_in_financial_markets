"""
eda_analyzer.py
---------------
Responsabilité unique : analyser et visualiser les données brutes.
Ne modifie aucune donnée. Produit uniquement des statistiques et graphiques.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


class EDAAnalyzer:
    """
    Analyse exploratoire des données du challenge CFM.

    Parameters
    ----------
    x_train : pd.DataFrame
        Données d'entrée d'entraînement.
    y_train : pd.DataFrame
        Cible d'entraînement (colonnes : ID, TARGET).
    output_dir : str
        Dossier où sauvegarder les graphiques.
    """

    VOLATILITY_PREFIX = "volatility"
    RETURN_PREFIX     = "return"

    def __init__(
        self,
        x_train: pd.DataFrame,
        y_train: pd.DataFrame,
        output_dir: str = "../outputs/"
    ) -> None:
        self.x_train    = x_train
        self.y_train    = y_train
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.volatility_columns = self._extract_columns(self.VOLATILITY_PREFIX)
        self.return_columns     = self._extract_columns(self.RETURN_PREFIX)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Lance l'ensemble de l'analyse exploratoire."""
        self.describe_dataset()
        self.analyze_missing_values()
        self.analyze_target_distribution()
        self.analyze_volatility_persistence()
        self.analyze_intraday_volatility_profile()

    def describe_dataset(self) -> None:
        """Affiche les dimensions et la composition du dataset."""
        n_rows, n_cols       = self.x_train.shape
        n_stocks             = self.x_train["product_id"].nunique()
        n_dates              = self.x_train["date"].nunique()
        n_volatility_cols    = len(self.volatility_columns)
        n_return_cols        = len(self.return_columns)

        print("=" * 55)
        print("DESCRIPTION DU DATASET")
        print("=" * 55)
        print(f"  Lignes totales        : {n_rows:>10,}")
        print(f"  Colonnes totales      : {n_cols:>10,}")
        print(f"  Stocks uniques        : {n_stocks:>10,}")
        print(f"  Jours uniques         : {n_dates:>10,}")
        print(f"  Colonnes volatilité   : {n_volatility_cols:>10,}")
        print(f"  Colonnes return dir.  : {n_return_cols:>10,}")
        print("=" * 55)

    def analyze_missing_values(self) -> None:
        """Analyse la structure des valeurs manquantes."""
        missing_per_column = self.x_train[self.volatility_columns].isnull().sum()
        missing_per_row    = self.x_train[self.volatility_columns].isnull().sum(axis=1)

        n_rows_with_missing = (missing_per_row > 0).sum()
        pct_rows_with_missing = 100 * n_rows_with_missing / len(self.x_train)

        print("\n" + "=" * 55)
        print("VALEURS MANQUANTES")
        print("=" * 55)
        print(f"  Lignes avec au moins 1 NaN : {n_rows_with_missing:,} "
              f"({pct_rows_with_missing:.1f}%)")
        print(f"  NaN moyen par ligne        : {missing_per_row.mean():.2f}")
        print(f"  Max NaN sur une ligne      : {missing_per_row.max()}")

        # Distribution des NaN par colonne (heure)
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.bar(range(len(missing_per_column)), missing_per_column.values, color="steelblue")
        ax.set_title("Nombre de NaN par barre de volatilité (9h30 → 13h55)", fontsize=13)
        ax.set_xlabel("Indice de la barre temporelle")
        ax.set_ylabel("Nombre de valeurs manquantes")
        ax.set_xticks(range(0, len(missing_per_column), 6))
        plt.tight_layout()
        plt.savefig(self.output_dir / "missing_values_by_timebar.png", dpi=150)
        plt.show()

    def analyze_target_distribution(self) -> None:
        """Analyse la distribution de la volatilité cible."""
        target = self.y_train["TARGET"]

        print("\n" + "=" * 55)
        print("DISTRIBUTION DE LA TARGET (vol 14h-16h)")
        print("=" * 55)
        print(target.describe().to_string())
        print(f"\n  Skewness  : {target.skew():.3f}")
        print(f"  Kurtosis  : {target.kurt():.3f}")

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Distribution brute
        axes[0].hist(target, bins=100, color="steelblue", edgecolor="none")
        axes[0].set_title("Distribution de la TARGET", fontsize=13)
        axes[0].set_xlabel("Volatilité réalisée (14h-16h)")
        axes[0].set_ylabel("Fréquence")

        # Distribution log-transformée
        axes[1].hist(np.log1p(target), bins=100, color="darkorange", edgecolor="none")
        axes[1].set_title("Distribution de log(1 + TARGET)", fontsize=13)
        axes[1].set_xlabel("log(1 + volatilité)")
        axes[1].set_ylabel("Fréquence")

        plt.tight_layout()
        plt.savefig(self.output_dir / "target_distribution.png", dpi=150)
        plt.show()

    def analyze_volatility_persistence(self) -> None:
        """
        Mesure la corrélation entre la volatilité moyenne du matin
        et la volatilité cible de l'après-midi.
        C'est la baseline économique fondamentale de ce problème.
        """
        mean_morning_vol = self.x_train[self.volatility_columns].mean(axis=1)
        target           = self.y_train["TARGET"]

        correlation = np.corrcoef(mean_morning_vol, target)[0, 1]

        print("\n" + "=" * 55)
        print("PERSISTANCE DE LA VOLATILITÉ")
        print("=" * 55)
        print(f"  Corrélation vol matin / TARGET : {correlation:.4f}")

        # Scatter plot (échantillon pour lisibilité)
        sample_idx = np.random.choice(len(target), size=min(10_000, len(target)), replace=False)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(
            mean_morning_vol.iloc[sample_idx],
            target.iloc[sample_idx],
            alpha=0.2, s=5, color="steelblue"
        )
        ax.set_title(f"Vol matin vs TARGET  (r = {correlation:.3f})", fontsize=13)
        ax.set_xlabel("Volatilité moyenne matin (9h30-13h55)")
        ax.set_ylabel("Volatilité cible (14h-16h)")
        plt.tight_layout()
        plt.savefig(self.output_dir / "volatility_persistence.png", dpi=150)
        plt.show()

    def analyze_intraday_volatility_profile(self) -> None:
        """
        Analyse le profil intraday moyen de la volatilité.
        Révèle les patterns en U caractéristiques des marchés actions.
        """
        mean_vol_by_bar = self.x_train[self.volatility_columns].mean()

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(range(len(mean_vol_by_bar)), mean_vol_by_bar.values,
                color="steelblue", linewidth=2)
        ax.set_title("Profil intraday moyen de la volatilité (9h30 → 13h55)", fontsize=13)
        ax.set_xlabel("Barre temporelle (0 = 9h30, 53 = 13h55)")
        ax.set_ylabel("Volatilité moyenne")
        ax.set_xticks(range(0, 54, 6))
        ax.set_xticklabels(
            [self.volatility_columns[i].split(" ")[1][:5]
             for i in range(0, 54, 6)],
            rotation=45
        )
        plt.tight_layout()
        plt.savefig(self.output_dir / "intraday_volatility_profile.png", dpi=150)
        plt.show()

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _extract_columns(self, prefix: str) -> list[str]:
        """Extrait les colonnes correspondant à un préfixe donné."""
        return [col for col in self.x_train.columns if col.startswith(prefix)]