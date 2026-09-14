import sys
import pandas as pd
import numpy as np
from numpy.typing import NDArray
from config import THETA_FILE, LEARNING_RATE, N_ITERATION
from src.helpers.utils import load_csv_dataset
from src.logistic_regression.logistic_regression import gradient_descent
from src.logistic_regression.plot_data import plot_dataset, plot_cost_history
from src.helpers.statistics import (
    ft_max,
    ft_mean,
    ft_min,
    ft_percentile,
    ft_std
)


def get_features(
        dataset: pd.DataFrame
    ) -> list[str]:

    features = []

    for column_name, column in dataset.items():
        if column.dtype == float:
            features.append(column_name)

    return features


def get_number_of_features(
        dataset: pd.DataFrame
    ) -> int:

    features = get_features(dataset)

    return len(features)


def get_targets(
        dataset: pd.DataFrame
    ) -> list[str]:

    labels = dataset["Hogwarts House"]
    targets = np.unique(labels)

    return targets


def initialize_theta(
        n: int
    ) -> NDArray[np.float64]:

    return np.zeros((
        n + 1,
        1
    ))


def prepare_training_data(
        dataset: pd.DataFrame,
        hogwarts_house: str
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:

    required_column = {"Hogwarts House"}
    if not required_column.issubset(dataset.columns):
        raise ValueError("dataset must contain 'Hogwarts House' column.")

    features = get_features(dataset)
    x = dataset[features].to_numpy()

    mask = dataset["Hogwarts House"] == hogwarts_house
    y = mask.to_numpy(dtype=np.float64)
    y = y.reshape(y.shape[0], 1)

    return x, y


def train_model(
        x: NDArray[np.float64],
        y: NDArray[np.float64],
        theta: NDArray[np.float64],
        n: int
    ) -> tuple[NDArray[np.float64], list[float]]:

    mean: float = x.mean()
    std: float = x.std()

    # standardisation (z-score)
    X = np.hstack((
        # (x - mean) / std,
        x,
        np.ones((x.shape[0], 1))
    ))

    print(f"X shape: {X.shape}\nY shape: {y.shape}\ntheta shape: {theta.shape}")
    final_theta, cost_history = gradient_descent(
        X,
        y,
        theta,
        learning_rate=LEARNING_RATE,
        n_iteration=N_ITERATION
    )

    # y = theta0'x' + theta1'
    #   -> y = theta0'((x - mean) / std) + theta1'
    #   -> y = (theta0' / std)x - (theta0'mean / std) + theta1'
    #   -> y = (theta0' / std)x + (theta1' - (theta0'mean / std))   => y = ax + b
    # a = theta0' / std
    # b = theta1' - (theta0'mean / std)

    return final_theta, cost_history


def save_theta_in_file(
        theta: NDArray[np.float64]
    ) -> None:

    with THETA_FILE.open("w") as f:
        for value in theta:
            f.write(f"{value}\n")


if __name__ == "__main__":

    try:
        if len(sys.argv) != 2:
            print(f"Usage: {sys.argv[0]} <dataset>.")
            sys.exit(1)

        dataset: pd.DataFrame = load_csv_dataset(sys.argv[1])
        if dataset is None:
            sys.exit(1)

        number_of_examples: int = dataset.shape[0]
        number_of_features = get_number_of_features(dataset)

        targets = get_targets(dataset)

        for target in targets:
            x, y = prepare_training_data(dataset, target)
            initial_theta = initialize_theta(number_of_features)

            final_theta, cost_history = train_model(x, y, initial_theta, number_of_features)

            # plot_cost_history(cost_history)

            save_theta_in_file(final_theta)

    except Exception as error:
        print("Program exited with a fatal error.", error)
