import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import COLORMAP
from src.helpers.data import get_features, load_csv_dataset
from src.helpers.statistics import ft_max, ft_mean, ft_min, ft_std


BIN_COUNT = 20
GRID_COLUMNS = 4


def get_feature_bounds(
        values: np.ndarray
    ) -> tuple[float, float] | None:
    """Bornes min/max d'une matière, toutes maisons confondues.

    Ces bornes servent à imposer le MÊME découpage en barres aux quatre
    maisons. Sans elles, matplotlib recalcule les bornes à chaque appel
    de hist() sur les seules données de la maison tracée : les quatre
    découpages diffèrent, les barres ne s'alignent plus, et la
    comparaison visuelle devient fausse.

    On écarte les NaN avant le calcul : une comparaison avec NaN est
    toujours fausse, donc ft_min renverrait sa valeur de départ si la
    première note était manquante.

    ft_min / ft_max (nos implémentations) plutôt que les builtins, pour
    rester cohérents avec le reste du projet.
    """

    known_values = list(values[~np.isnan(values)])
    if not known_values:
        return None

    return ft_min(known_values), ft_max(known_values)


def plot_feature_histogram(
        axis,
        values: np.ndarray,
        labels: np.ndarray,
        bounds: tuple[float, float]
    ) -> None:
    """Superpose les distributions des quatre maisons pour une matière."""

    for house, color in COLORMAP.items():
        # Un élève est retenu s'il est de cette maison ET a bien une note.
        mask = (labels == house) & ~np.isnan(values)

        axis.hist(
            values[mask],
            bins=BIN_COUNT,
            range=bounds,
            # density=True normalise l'aire de chaque histogramme à 1.
            # Indispensable ici : les maisons sont déséquilibrées
            # (529 Hufflepuff contre 301 Slytherin). Sans ça, on
            # comparerait des tailles de maison au lieu de comparer des
            # distributions de notes.
            density=True,
            color=color,
            alpha=0.5,
            label=house
        )


def get_homogeneity_score(
        values: np.ndarray,
        labels: np.ndarray
    ) -> float | None:
    """Mesure à quel point les quatre maisons se ressemblent.

    score = écart-type des 4 moyennes de maison
            / écart-type de toutes les notes de la matière

    Le numérateur mesure la séparation des maisons, le dénominateur la
    dispersion naturelle de la matière. Leur rapport est sans dimension :
    c'est ce qui rend les 13 matières comparables entre elles, alors que
    leurs échelles n'ont rien à voir (Arithmancy se compte en dizaines de
    milliers, Care of Magical Creatures en unités). Comparer des écarts
    de moyennes bruts ne mesurerait que l'unité de notation.

    Score proche de 0 : les maisons ont la même moyenne -> homogène.
    Score élevé : les maisons sont séparées -> matière discriminante.
    """

    house_means = []
    for house in COLORMAP:
        mask = (labels == house) & ~np.isnan(values)
        house_values = list(values[mask])
        if not house_values:
            return None
        house_means.append(ft_mean(house_values))

    known_values = list(values[~np.isnan(values)])
    overall_std = ft_std(known_values)
    if not overall_std:
        return None

    return ft_std(house_means) / overall_std


def print_homogeneity_ranking(
        data: dict[str, np.ndarray],
        features: list[str],
        labels: np.ndarray
    ) -> None:
    """Classe les matières de la plus homogène à la plus discriminante."""

    scores = []
    for feature in features:
        score = get_homogeneity_score(data[feature], labels)
        if score is not None:
            scores.append((score, feature))

    scores.sort()

    print("\nHomogénéité des matières entre les quatre maisons")
    print("(écart-type des moyennes de maison / écart-type global)")
    print("Plus le score est bas, plus la matière est homogène.\n")

    for rank, (score, feature) in enumerate(scores, start=1):
        print(f"{rank:>3}. {feature:<32} {score:.4f}")

    print()


def create_histogram_grid(
        data: dict[str, np.ndarray],
        features: list[str],
        labels: np.ndarray
    ) -> None:
    """Affiche une grille d'histogrammes, une case par matière."""

    row_count = (len(features) + GRID_COLUMNS - 1) // GRID_COLUMNS

    figure, axes = plt.subplots(
        nrows=row_count,
        ncols=GRID_COLUMNS,
        figsize=(16, 10)
    )
    flat_axes = axes.flatten()

    for index, feature in enumerate(features):
        axis = flat_axes[index]
        values = data[feature]

        bounds = get_feature_bounds(values)
        if bounds is None:
            axis.axis("off")
            continue

        plot_feature_histogram(axis, values, labels, bounds)

        axis.set_title(feature, fontsize=9)
        # Les valeurs absolues n'apportent rien : ce qu'on lit ici, c'est
        # le degré de superposition des quatre courbes.
        axis.set_xticks([])
        axis.set_yticks([])

    # Les matières ne remplissent pas forcément la grille (13 sur 16).
    for axis in flat_axes[len(features):]:
        axis.axis("off")

    handles, legend_labels = flat_axes[0].get_legend_handles_labels()
    figure.legend(handles, legend_labels, loc="lower right", fontsize=11)

    figure.suptitle(
        "Distribution des notes par maison\n"
        "La matière homogène est celle dont les quatre courbes se "
        "superposent le mieux",
        fontsize=13
    )

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

        # Affiche avant d'ouvrir la fenêtre : plt.show() bloque le
        # programme jusqu'à ce qu'elle soit fermée.
        print_homogeneity_ranking(data, features, labels)

        create_histogram_grid(data, features, labels)

    except Exception as error:
        print("Program exited with a fatal error.", error)
