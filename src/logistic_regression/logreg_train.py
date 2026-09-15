import sys
import pandas as pd
import numpy as np
from numpy.typing import NDArray
from config import THETA_FILE, LEARNING_RATE, N_ITERATION
from src.helpers.utils import load_csv_dataset
from src.logistic_regression.logistic_regression import gradient_descent


def get_features(
        dataset: pd.DataFrame
    ) -> list[str]:

    features = []

    for column_name, column in dataset.items():
        if column.dtype == float:
            features.append(column_name)

    return features


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

    mask = dataset["Hogwarts House"] == hogwarts_house
    y = mask.to_numpy(dtype=np.float64)
    y = y.reshape(y.shape[0], 1)

    return y


def train_model(
        x: NDArray[np.float64],
        y: NDArray[np.float64],
        theta: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], list[float]]:

    mean: NDArray[np.float64] = x.mean(axis=0)
    std: NDArray[np.float64] = x.std(axis=0)

    # standardisation (z-score)
    X = np.hstack((
        (x - mean) / std,
        np.ones((x.shape[0], 1))
    ))

    # print(f"X shape: {X.shape}\nY shape: {y.shape}\ntheta shape: {theta.shape}")
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
        features: list[str],
        hogwarts_house: str,
        theta: NDArray[np.float64]
    ) -> None:

    header = ["House", *features, "Bias"]
    if not THETA_FILE.exists():
        with THETA_FILE.open("w") as f:
            f.write(",".join(header) + "\n")
    with THETA_FILE.open("a") as f:
        values = [hogwarts_house, *(str(value.item()) for value in theta)]
        f.write(",".join(values) + "\n")


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

    targets: list[str] = get_targets(dataset)
    features: list[str] = get_features(dataset)

    x = dataset[features].copy()
    x = x.fillna(x.mean())
    x = x.to_numpy()

    for hogwarts_house in targets:
        y = prepare_training_data(dataset, hogwarts_house)
        initial_theta = initialize_theta(len(features))

        final_theta, cost_history = train_model(x, y, initial_theta)

        save_theta_in_file(features, hogwarts_house, final_theta)

    return 0


if __name__ == "__main__":

    sys.exit(main(len(sys.argv), sys.argv))
