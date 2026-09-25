import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import COLORMAP
from src.helpers.data import get_features, load_csv_dataset
from src.helpers.statistics import ft_mean, ft_std


RANKING_SIZE = 5


def ft_correlation(
        values_1: np.ndarray,
        values_2: np.ndarray
) -> float | None:
    """Corrélation de Pearson : covariance / (écart-type × écart-type).

    Sans dimension, entre -1 et 1. |r| = 1 : points parfaitement alignés.
    """

    # Une paire n'est exploitable que si l'élève a ses DEUX notes.
    known = ~np.isnan(values_1) & ~np.isnan(values_2)
    x = list(values_1[known])
    y = list(values_2[known])
    if len(x) < 2:
        return None

    mean_x = ft_mean(x)
    mean_y = ft_mean(y)

    covariance = 0
    for index in range(len(x)):
        covariance += (x[index] - mean_x) * (y[index] - mean_y)
    covariance /= len(x) - 1

    std_x = ft_std(x)
    std_y = ft_std(y)
    if not std_x or not std_y:
        return None

    return covariance / (std_x * std_y)


def get_correlated_pairs(
        data: dict[str, np.ndarray],
        features: list[str]
) -> list[tuple[float, float, str, str]]:
    """Toutes les paires, triées par similarité décroissante.

    Tri sur |r| : r = -1 est aussi similaire que r = +1, au signe près.
    """

    pairs = []

    for i in range(len(features)):
        # j > i : chaque paire une seule fois, jamais avec elle-même.
        for j in range(i + 1, len(features)):
            feature_1 = features[i]
            feature_2 = features[j]

            correlation = ft_correlation(data[feature_1], data[feature_2])
            if correlation is not None:
                pairs.append(
                    (abs(correlation), correlation, feature_1, feature_2)
                )

    pairs.sort(reverse=True)

    return pairs


def print_correlation_ranking(
        pairs: list[tuple[float, float, str, str]]
) -> None:
    """Affiche les couples de matières les plus similaires."""

    print("\nCouples de matières les plus similaires")
    print("(coefficient de corrélation de Pearson)")
    print("|r| = 1 : points parfaitement alignés sur une droite.\n")

    for rank, pair in enumerate(pairs[:RANKING_SIZE], start=1):
        _, correlation, feature_1, feature_2 = pair
        print(f"{rank:>3}. {feature_1:<30} / {feature_2:<30}"
              f"  r = {correlation:+.4f}")

    print()


def create_scatter_plot(
        data: dict[str, np.ndarray],
        feature_1: str,
        feature_2: str,
        labels: np.ndarray,
        correlation: float
) -> None:
    """Nuage de points du couple de matières le plus similaire."""

    values_1 = data[feature_1]
    values_2 = data[feature_2]

    figure, axis = plt.subplots(figsize=(10, 8))

    for house, color in COLORMAP.items():
        mask = (
            (labels == house)
            & ~np.isnan(values_1)
            & ~np.isnan(values_2)
        )
        axis.scatter(
            values_1[mask],
            values_2[mask],
            color=color,
            label=house,
            s=8,
            alpha=0.6
        )

    axis.set_xlabel(feature_1)
    axis.set_ylabel(feature_2)
    axis.set_title(
        f"Les deux matières les plus similaires  (r = {correlation:+.4f})",
        fontsize=13
    )
    axis.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    try:
        if len(sys.argv) != 2:
            print(f"Usage: {sys.argv[0]} <dataset>.")
            sys.exit(1)

        dataset: pd.DataFrame = load_csv_dataset(sys.argv[1])
        if dataset is None:
            sys.exit(1)

        features = get_features(dataset)
        data = {
            feature: np.array(dataset[feature])
            for feature in features
        }
        labels = np.array(dataset["Hogwarts House"])

        pairs = get_correlated_pairs(data, features)
        if not pairs:
            print("Aucune paire de matières exploitable.")
            sys.exit(1)

        # Affiché avant plt.show(), qui bloque jusqu'à fermeture.
        print_correlation_ranking(pairs)

        _, correlation, feature_1, feature_2 = pairs[0]
        create_scatter_plot(
            data,
            feature_1,
            feature_2,
            labels,
            correlation
        )

    except Exception as error:
        print("Program exited with a fatal error.", error)
