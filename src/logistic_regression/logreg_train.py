import sys
import pandas as pd
import numpy as np
from numpy.typing import NDArray
from src.helpers.data import (
    load_csv_dataset,
    get_features,
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

    required_column = { "Hogwarts House" }
    if not required_column.issubset(dataset.columns):
        print("Value error: Dataset must contain 'Hogwarts House' column.")
        return 1

    targets: list[str] = get_targets(dataset)
    features: list[str] = get_features(dataset)
    x: NDArray[np.float64] = get_clean_data(dataset, features)
    x, mean, std = standardize_data(x)

    initialize_theta_file(features)

    for hogwarts_house in targets:
        y: NDArray[np.float64] = get_binary_target(dataset, hogwarts_house)
        initial_theta: NDArray[np.float64] = initialize_theta(len(features))

        standardized_theta, cost_history = train_model(
            x,
            y,
            initial_theta
        )

        final_theta = destandardize_theta(
            standardized_theta,
            mean,
            std
        )
        save_theta_in_file(
            hogwarts_house,
            final_theta
        )

    return 0


if __name__ == "__main__":

    sys.exit(main(len(sys.argv), sys.argv))
