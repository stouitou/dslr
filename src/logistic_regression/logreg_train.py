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
    destandardize_theta,
    train_model
)
from src.helpers.output import (
    initialize_theta_file,
    save_theta_in_file
)
from src.helpers.plotting import plot_cost_history
from src.helpers.confusion_matrix import plot_confusion_matrix


def main(
        argc: int,
        argv: list[str]
) -> int:

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

    for hogwarts_house in targets:
        y: NDArray[np.float64] = get_binary_target(dataset, hogwarts_house)
        initial_theta: NDArray[np.float64] = initialize_theta(
            len(RELEVANT_FEATURES)
        )

        standardized_theta, history = train_model(
            x,
            y,
            initial_theta
        )

        cost_history[hogwarts_house] = history

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
    plot_confusion_matrix(dataset, targets)

    return 0


if __name__ == "__main__":

    sys.exit(main(len(sys.argv), sys.argv))
