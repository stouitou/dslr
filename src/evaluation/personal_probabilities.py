import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from matplotlib import axes
from matplotlib.ticker import PercentFormatter
from numpy.typing import NDArray
from pathlib import Path
from src.helpers.data import (
    load_csv_dataset,
    prepare_prediction_data
)
from src.helpers.logistic_regression import model


MAX_PROBABILITY_GAP = 0.5
CONFIDENCE_THRESHOLD = 0.5


def get_probability(
        x: NDArray[np.float64],
        theta: dict[str, NDArray[np.float64]]
) -> dict[str, float]:

    probabilities: dict[str, float] = {}

    for hogwarts_house in theta:
        probabilities[hogwarts_house] = float(
            model(x, theta[hogwarts_house])[0]
        )

    return probabilities


def is_ambiguous(
        probabilities: dict[str, float]
) -> bool:

    values = sorted(probabilities.values(), reverse=True)

    best_probability = values[0]
    second_best = values[1]

    difference = best_probability - second_best

    return (
        difference < MAX_PROBABILITY_GAP
        or best_probability < CONFIDENCE_THRESHOLD
    )


def get_ambiguous_students(
        X: NDArray[np.float64],
        theta: dict[str, NDArray[np.float64]]
) -> dict[int, dict[str, float]]:

    ambiguous_students: dict[int, dict[str, float]] = {}

    for i in range(X.shape[0]):
        probabilities = get_probability(X[i], theta)

        if is_ambiguous(probabilities):
            ambiguous_students[i] = probabilities

    return ambiguous_students


def plot_student_probability(
        ax: axes,
        student_id: int,
        probabilities: dict[str, float],
        short_house_names: list[str]
) -> None:

    bars = ax.bar(
        probabilities.keys(),
        probabilities.values()
    )

    ax.set_ylim(0, 1.02)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xticklabels(
        short_house_names,
        rotation=45,
        ha="right"
    )
    ax.set_title(f"Student {student_id}")

    for bar, probability in zip(bars, probabilities.values()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{probability:.1%}",
            ha="center",
            va="bottom",
            fontsize=8
        )


def display_probability(
        ambiguous_students: dict[int, dict[str, float]]
) -> None:

    ncols = 5
    nrows = (len(ambiguous_students) + ncols - 1) // ncols

    house_names = list(next(iter(ambiguous_students.values())).keys())
    short_house_names = [
        hogwarts_house[:3]
        for hogwarts_house
        in house_names
    ]

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(20, 4 * nrows)
    )
    axes = np.asarray(axes).reshape(nrows, ncols)

    for index, (id, probabilities) in enumerate(ambiguous_students.items()):
        row = index // ncols
        col = index % ncols
        ax = axes[row, col]

        plot_student_probability(
            ax,
            id,
            probabilities,
            short_house_names
        )

        if col != 0:
            ax.set_yticklabels([])

    for index in range(len(ambiguous_students), nrows * ncols):
        fig.delaxes(axes.flat[index])

    fig.supylabel("Probability")

    plt.subplots_adjust(
        left=0.05,
        right=0.98,
        bottom=0.15,
        top=0.92,
        wspace=0.15,
        hspace=0.35
    )

    plt.show()


def main(
        argc: int,
        argv: list[str]
) -> int:

    if argc != 3:
        print(f"Usage: {argv[0]} <dataset> <theta_file>.")
        return 1

    dataset: pd.DataFrame | None = load_csv_dataset(argv[1])
    if dataset is None:
        return 1

    X, theta = prepare_prediction_data(
        dataset,
        Path(argv[2])
    )
    if theta is None:
        return 1

    ambiguous_students = get_ambiguous_students(X, theta)

    display_probability(ambiguous_students)

    return 0


if __name__ == "__main__":

    sys.exit(main(len(sys.argv), sys.argv))
