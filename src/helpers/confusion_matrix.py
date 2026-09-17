import matplotlib.pyplot as plt
import pandas as pd
from config import RESULT_FILE
from src.helpers.data import load_csv_dataset


def get_confusion_matrix_data(
        dataset: pd.DataFrame
) -> tuple[list[str], list[str]]:

    target = "Hogwarts House"

    truth = dataset[target].to_numpy()

    result: pd.DataFrame = load_csv_dataset(RESULT_FILE)
    prediction = result[target].to_numpy()

    return truth, prediction


def initiate_confusion_matrix(
        targets: list[str]
) -> dict[str, dict[str, int]]:

    matrix = {}

    for hogwarts_house1 in targets:
        matrix[hogwarts_house1] = {}
        for hogwarts_house2 in targets:
            matrix[hogwarts_house1][hogwarts_house2] = 0

    return matrix


def plot_confusion_matrix(
        dataset: pd.DataFrame,
        targets: list[str]
) -> None:

    y_true, y_predicted = get_confusion_matrix_data(dataset)
    matrix = initiate_confusion_matrix(targets)

    for i in range(len(y_true)):
        matrix[y_true[i]][y_predicted[i]] += 1

    values1 = list(matrix.values())
    values2 = [list(value.values()) for value in values1]
    confusion_matrix = values2

    fig, ax = plt.subplots()
    ax.imshow(confusion_matrix)
    plt.title("Confusion Matrix")

    plt.show()
