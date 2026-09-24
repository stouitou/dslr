import numpy as np
import pandas as pd
import sys
from numpy.typing import NDArray
from config import RELEVANT_FEATURES
from src.helpers.data import (
    load_csv_dataset,
    get_targets,
    get_clean_data,
    standardize_data,
    get_binary_target
)
from src.helpers.logistic_regression import (
    initialize_theta,
    destandardize_theta
)
from src.helpers.plotting import plot_cost_history
from src.bonus_SGD.stochastic_gradient_descent import train_model_sgd
from src.bonus_SGD.output_sgd import (
    initialize_theta_file,
    save_theta_in_file
)


def main(
        argc: int,
        argv: list[str]
) -> int:
    """Entraîne les 4 classifieurs one-vs-all par descente stochastique.

    Bonus de logreg_train.py : seule la descente change.
    """

    if argc != 2:
        print(f"Usage: {argv[0]} <dataset>.")
        return 1

    dataset: pd.DataFrame | None = load_csv_dataset(argv[1])
    if dataset is None:
        return 1

    required_column = {"Hogwarts House"}
    if not required_column.issubset(dataset.columns):
        print("Value error: Dataset must contain 'Hogwarts House' column.")
        return 1

    targets: list[str] = get_targets(dataset)
    x: NDArray[np.float64] = get_clean_data(dataset, RELEVANT_FEATURES)
    x, mean, std = standardize_data(x)

    initialize_theta_file(RELEVANT_FEATURES)

    cost_history: dict[str, list[float]] = {}

    # One-vs-all : un classifieur binaire par maison.
    for hogwarts_house in targets:
        y: NDArray[np.float64] = get_binary_target(dataset, hogwarts_house)
        initial_theta: NDArray[np.float64] = initialize_theta(
            len(RELEVANT_FEATURES)
        )

        standardized_theta, history = train_model_sgd(
            x,
            y,
            initial_theta
        )

        cost_history[hogwarts_house] = history

        # Appris sur données standardisées : on ramène les poids à
        # l'échelle d'origine pour les appliquer au dataset brut.
        final_theta = destandardize_theta(
            standardized_theta,
            mean,
            std
        )
        save_theta_in_file(
            hogwarts_house,
            final_theta
        )

    plot_cost_history(cost_history)

    return 0


if __name__ == "__main__":

    sys.exit(main(len(sys.argv), sys.argv))
