# Duplique src/helpers/output.py pour écrire dans des fichiers distincts :
# le bonus ne doit jamais écraser les sorties de la partie obligatoire.

import numpy as np
from numpy.typing import NDArray
from config import RESULT_SGD_FILE, THETA_SGD_FILE


def initialize_theta_file(
        features: list[str]
) -> None:
    """Crée THETA_SGD_FILE avec sa ligne d'en-tête."""

    with THETA_SGD_FILE.open("w") as file:
        header = [
            "House",
            *features,
            "Bias"
        ]
        file.write(",".join(header) + "\n")


def save_theta_in_file(
        hogwarts_house: str,
        theta: NDArray[np.float64]
) -> None:
    """Ajoute à THETA_SGD_FILE les poids d'une maison."""

    with THETA_SGD_FILE.open("a") as f:
        values: list[str] = [
            hogwarts_house, *(str(value) for value in theta)
        ]
        f.write(",".join(values) + "\n")


def initialize_result_file(
) -> None:
    """Crée RESULT_SGD_FILE avec sa ligne d'en-tête."""

    with RESULT_SGD_FILE.open("w") as file:
        header = [
            "Index",
            "Hogwarts House"
        ]
        file.write(",".join(header) + "\n")


def save_result_in_file(
        index: int,
        hogwarts_house: str,
) -> None:
    """Ajoute à RESULT_SGD_FILE la prédiction d'un élève."""

    with RESULT_SGD_FILE.open("a") as f:
        values: list[str] = [
            str(index),
            hogwarts_house
        ]
        f.write(",".join(values) + "\n")
