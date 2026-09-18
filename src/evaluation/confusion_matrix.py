import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from matplotlib import axes
from matplotlib.colors import LogNorm
from numpy.typing import NDArray
from src.helpers.data import (
    get_targets,
    load_csv_dataset
)


def get_confusion_matrix_data(
        dataset_truth: pd.DataFrame,
        dataset_prediction: pd.DataFrame,
        targets: list[str]
) -> tuple[NDArray[np.str_], NDArray[np.str_]]:

    target = "Hogwarts House"

    if target not in dataset_truth.columns:
        raise ValueError(
            f"'{target}' column is missing from truth dataset."
        )

    if dataset_truth[target].isna().any():
        raise ValueError(
            f"One or more values in the '{target}' column are missing."
        )

    if target not in dataset_prediction.columns:
        raise ValueError(
            f"'{target}' column is missing from prediction dataset."
        )

    truth = dataset_truth[target].to_numpy()
    prediction = dataset_prediction[target].to_numpy()

    if len(truth) != len(prediction):
        raise ValueError(
            "Truth and prediction datasets have different numbers of rows."
        )

    invalid_predictions = set(prediction) - set(targets)
    if invalid_predictions:
        raise ValueError(
            f"Unknown prediction classes: {invalid_predictions}"
        )

    return truth, prediction


def initiate_confusion_matrix(
        targets: list[str]
) -> dict[str, dict[str, int]]:

    matrix = {}

    for true_house in targets:
        matrix[true_house] = {}
        for predicted_house in targets:
            matrix[true_house][predicted_house] = 0

    return matrix


def add_matrix_values(
        ax: axes,
        confusion_matrix: list[list[int]]
) -> None:

    for i in range(len(confusion_matrix)):
        for j in range(len(confusion_matrix)):
            value = confusion_matrix[i][j]
            text_color = "white" if value > 100 else "black"

            ax.text(
                j,
                i,
                value,
                ha="center",
                va="center",
                color=text_color
            )


def set_matrix_labels(
        ax: axes,
        targets: list[str]
) -> None:

    ax.set_xticks(range(len(targets)))
    ax.set_yticks(range(len(targets)))

    ax.set_xticklabels(targets)
    ax.set_yticklabels(targets)
    ax.xaxis.tick_top()

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.xaxis.set_label_position("top")


def plot_confusion_matrix(
        confusion_matrix: list[list[int]],
        targets: list[str]
) -> None:

    fig, ax = plt.subplots()
    ax.imshow(
        confusion_matrix,
        cmap="Blues",
        norm=LogNorm(
            vmin=1,
            vmax=max(map(max, confusion_matrix))
        )
    )

    add_matrix_values(ax, confusion_matrix)
    set_matrix_labels(ax, targets)

    fig.text(
        0.5,
        0.05,
        "Confusion Matrix",
        ha="center",
        fontsize=16
    )

    plt.show()


def display_confusion_matrix(
        dataset_truth: pd.DataFrame,
        dataset_prediction: pd.DataFrame,
        targets: list[str]
) -> None:

    y_true, y_predicted = get_confusion_matrix_data(
        dataset_truth,
        dataset_prediction,
        targets
    )

    matrix_dictionary = initiate_confusion_matrix(targets)
    for i in range(len(y_true)):
        matrix_dictionary[y_true[i]][y_predicted[i]] += 1
    confusion_matrix: list[list[int]] = [
        list(value.values())
        for value
        in list(matrix_dictionary.values())
    ]

    plot_confusion_matrix(confusion_matrix, targets)


def main(
        argc: int,
        argv: list[str]
) -> int:

    if argc != 3:
        print(f"Usage: {argv[0]} <dataset> <houses_file>.")
        return 1

    dataset_truth: pd.DataFrame | None = load_csv_dataset(argv[1])
    if dataset_truth is None:
        return 1

    dataset_prediction: pd.DataFrame | None = load_csv_dataset(argv[2])
    if dataset_prediction is None:
        return 1

    targets: list[str] = get_targets(dataset_truth)

    try:
        display_confusion_matrix(
            dataset_truth,
            dataset_prediction,
            targets
        )

    except ValueError as error:
        print(f"Value error: {error}")
        return 1

    return 0


if __name__ == "__main__":

    sys.exit(main(len(sys.argv), sys.argv))
