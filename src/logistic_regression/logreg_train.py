import sys
import pandas as pd
import numpy as np
from numpy.typing import NDArray
from config import THETA_FILE, LEARNING_RATE, N_ITERATION
from src.helpers.utils import load_csv_dataset
from src.logistic_regression.logistic_regression import gradient_descent
from src.logistic_regression.plot_data import plot_cost_history


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


def prepare_training_data(
        dataset: pd.DataFrame,
        hogwarts_house: str
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:

    required_column = {"Hogwarts House"}
    if not required_column.issubset(dataset.columns):
        raise ValueError("dataset must contain 'Hogwarts House' column.")

    x: NDArray[np.float64] = dataset[features].to_numpy()
    mask = dataset["Hogwarts House"] == hogwarts_house
    y: NDArray[np.float64] = mask.to_numpy(dtype=np.float64).reshape(mask.shape[0], 1)
    print(y)

    return x, y


def train_model(
        x: NDArray[np.float64],
        y: NDArray[np.float64],
        theta: NDArray[np.float64]
    ) -> tuple[float, float, list[float]]:

    mean: float = x.mean()
    std: float = x.std()

    # standardisation (z-score)
    X: NDArray[np.float64] = np.hstack((
        (x - mean) / std,
        np.ones((x.shape[0], 1))
    ))

    print(f"x shape: {X.shape}\ny shape: {y.shape}\ntheta shape: {theta.shape}")
    theta_final, cost_history = gradient_descent(
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
    theta0: float = theta_final[0, 0] / std
    theta1: float = theta_final[1, 0] - theta_final[0, 0] * mean / std

    return theta0, theta1, cost_history


def save_theta_in_file(
        theta0: float,
        theta1: float
    ) -> None:

    with THETA_FILE.open("w") as f:
        f.write(f"{theta0}\n")
        f.write(f"{theta1}\n")


if __name__ == "__main__":

    try:
        if len(sys.argv) != 2:
            print(f"Usage: {sys.argv[0]} <dataset>.")
            sys.exit(1)

        dataset: pd.DataFrame = load_csv_dataset(sys.argv[1])
        if dataset is None:
            sys.exit(1)

        number_of_examples: int = dataset.shape[0]
        features = get_features(dataset)
        number_of_features: int = len(features)

        targets = get_targets(dataset)
        for hogwarts_house in targets:
            print("Prepare training data...")
            x, y = prepare_training_data(dataset, hogwarts_house)

            print("Train model...")
            theta: NDArray[np.float64] = np.zeros((
                number_of_features + 1,
                1
            ))
            theta0, theta1, cost_history = train_model(x, y, theta)

        # plot_cost_history(cost_history)

        save_theta_in_file(theta0, theta1)

    except Exception as error:
        print("Program exited with a fatal error.", error)
