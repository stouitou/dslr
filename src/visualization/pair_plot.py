import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.helpers.utils import load_csv_dataset


def get_features(
        dataset: pd.DataFrame
    ) -> list[str]:

    features = []

    for column_name, column in dataset.items():
        if column.dtype == float:
            features.append(column_name)

    return features


def plot_histogram(
        axis,
        X: np.ndarray,
        labels: np.ndarray,
        colormap: dict[str, str]
    ) -> None:

    for label in colormap:
        mask = labels == label
        axis.hist(
            X[mask],
            color=colormap[label],
            bins=15,
            alpha=0.6
            )


def plot_scatter(
        axis,
        X: np.ndarray,
        y: np.ndarray,
        labels: np.ndarray,
        colormap: dict[str, str]
    ) -> None:

    for label in colormap:
        mask = labels == label
        axis.scatter(
            x=X[mask],
            y=y[mask],
            color=colormap[label],
            label=label,
            s=2,
            alpha=0.6
        )


def set_axis_labels(
        axis,
        feature_index_1: int,
        feature_index_2: int,
        features: list[str]
    ) -> None:

    if feature_index_1 == len(features) - 1:
        axis.set_xlabel(
            features[feature_index_2],
            fontsize=8,
            rotation=45,
        )

    if feature_index_2 == 0:
        axis.set_ylabel(
            features[feature_index_1],
            fontsize=8,
            rotation=45,
            ha="right"
        )


def plot_single_pair(
        axis,
        feature_index_1: int,
        feature_index_2: int,
        X: np.ndarray,
        y: np.ndarray,
        features: list[str],
        labels: np.ndarray,
        colormap: dict[str, str]
    ) -> None:

    # Plot distribution histogram if the features are the same (diagonal of the pair-plot).
    if feature_index_1 == feature_index_2:
        plot_histogram(
            axis,
            X,
            labels,
            colormap
        )
    else:
        plot_scatter(
            axis,
            X,
            y,
            labels,
            colormap
        )

    axis.set_xticks([])
    axis.set_yticks([])

    set_axis_labels(
        axis,
        feature_index_1,
        feature_index_2,
        features
    )


def create_pair_plot(
        data: dict[str, np.ndarray],
        features: list[str],
        labels: np.ndarray,
        colormap: dict[str, str]
    ) -> None:

    feature_count = len(features)

    fig, axis = plt.subplots(
        nrows=feature_count,
        ncols=feature_count,
        figsize=(15, 15)
    )

    for i in range(feature_count):
        for j in range(feature_count):
            X = data[features[i]]
            y = data[features[j]]

            plot_single_pair(
                axis[i, j],
                i,
                j,
                X,
                y,
                features,
                labels,
                colormap
            )

    plt.subplots_adjust(
        wspace=0.05,
        hspace=0.05
    )

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
        colormap = {
            "Gryffindor": "red",
            "Hufflepuff": "yellow",
            "Ravenclaw": "blue",
            "Slytherin": "green"
        }

        create_pair_plot(
            data,
            features,
            labels,
            colormap
        )

    except Exception as error:
        print("Program exited with a fatal error.", error)
