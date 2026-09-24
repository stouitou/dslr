import pandas as pd
import sys
from pathlib import Path
from src.helpers.data import (
    load_csv_dataset,
    prepare_prediction_data
)
from src.helpers.logistic_regression import model
from src.bonus_SGD.output_sgd import (
    initialize_result_file,
    save_result_in_file
)


def main(
        argc: int,
        argv: list[str]
) -> int:
    """Prédit les maisons à partir de poids entraînés par SGD.

    Bonus de logreg_predict.py : seul le fichier de sortie change.
    """

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

    initialize_result_file()

    # One-vs-all : on retient la maison de plus forte probabilité.
    for i in range(dataset.shape[0]):
        result: str = ""
        best_probability: float = 0
        for hogwarts_house in theta:
            probability = float(model(X[i], theta[hogwarts_house])[0])
            if probability > best_probability:
                best_probability = probability
                result = hogwarts_house

        save_result_in_file(i, result)

    return 0


if __name__ == "__main__":

    sys.exit(main(len(sys.argv), sys.argv))
