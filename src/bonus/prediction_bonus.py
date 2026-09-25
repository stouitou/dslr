import pandas as pd
from pathlib import Path
from src.helpers.data import (
    load_csv_dataset,
    prepare_prediction_data
)
from src.helpers.logistic_regression import model
from src.bonus.output_bonus import (
    initialize_result_file,
    save_result_in_file
)


def run_prediction(
        dataset_path: str,
        theta_path: str,
        result_file: Path
) -> int:
    """Prédit les maisons à partir d'un fichier de poids.

    Identique à logreg_predict.py : c'est le même modèle, seule la façon
    dont les poids ont été obtenus diffère. Seule la destination varie.
    """

    dataset: pd.DataFrame | None = load_csv_dataset(dataset_path)
    if dataset is None:
        return 1

    X, theta = prepare_prediction_data(dataset, Path(theta_path))
    if theta is None:
        return 1

    initialize_result_file(result_file)

    # One-vs-all : on retient la maison de plus forte probabilité.
    for i in range(dataset.shape[0]):
        result: str = ""
        best_probability: float = 0
        for hogwarts_house in theta:
            probability = float(model(X[i], theta[hogwarts_house])[0])
            if probability > best_probability:
                best_probability = probability
                result = hogwarts_house

        save_result_in_file(i, result, result_file)

    return 0
