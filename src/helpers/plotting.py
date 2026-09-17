import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import axes
from numpy.typing import NDArray
from src.helpers.data import get_features


def plot_histogram(
        ax: axes,
        x_values: np.ndarray,
        labels: np.ndarray,
        colormap: dict[str, str]
) -> None:

    for label in colormap:
        mask = labels == label
        ax.hist(
            x_values[mask],
            color=colormap[label],
            bins=15,
            alpha=0.6
            )


def plot_scatter(
        ax: axes,
        x_values: np.ndarray,
        y_values: np.ndarray,
        labels: np.ndarray,
        colormap: dict[str, str]
) -> None:

    for label in colormap:
        mask = labels == label
        ax.scatter(
            x=x_values[mask],
            y=y_values[mask],
            color=colormap[label],
            label=label,
            s=2,
            alpha=0.6
        )


def set_axis_labels(
        ax: axes,
        feature_index_1: int,
        feature_index_2: int,
        features: list[str]
) -> None:

    if feature_index_1 == len(features) - 1:
        ax.set_xlabel(
            features[feature_index_2],
            fontsize=8,
            rotation=45,
        )

    if feature_index_2 == 0:
        ax.set_ylabel(
            features[feature_index_1],
            fontsize=8,
            rotation=45,
            ha="right"
        )


def plot_single_pair(
        axis: axes,
        feature_index_1: int,
        feature_index_2: int,
        x_values: np.ndarray,
        y_values: np.ndarray,
        features: list[str],
        labels: np.ndarray,
        colormap: dict[str, str]
) -> None:

    if feature_index_1 == feature_index_2:
        plot_histogram(
            axis,
            x_values,
            labels,
            colormap
        )
    else:
        plot_scatter(
            axis,
            x_values,
            y_values,
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
            feature_x = data[features[i]]
            feature_y = data[features[j]]

            plot_single_pair(
                axis[i, j],
                i,
                j,
                feature_x,
                feature_y,
                features,
                labels,
                colormap
            )

    plt.subplots_adjust(
        wspace=0.05,
        hspace=0.05
    )

    plt.show()


def prepare_plot_data(
        dataset: pd.DataFrame
) -> tuple[
    dict[str, NDArray[np.float64]],
    list[str],
    NDArray[np.str_],
    dict[str, str]
]:

    features = get_features(dataset)

    data = {
        feature: dataset[feature].to_numpy()
        for feature in features
    }

    labels = dataset["Hogwarts House"].to_numpy()

    colormap = {
        "Gryffindor": "red",
        "Hufflepuff": "yellow",
        "Ravenclaw": "blue",
        "Slytherin": "green"
    }

    return data, features, labels, colormap
