# Duplique src/helpers/output.py en rendant le chemin paramétrable, pour que
# chaque bonus écrive dans ses propres fichiers sans jamais toucher à ceux
# de la partie obligatoire.

import numpy as np
from numpy.typing import NDArray
from pathlib import Path


def initialize_theta_file(
        features: list[str],
        theta_file: Path
) -> None:
    """Crée le fichier de poids avec sa ligne d'en-tête."""

    with theta_file.open("w") as file:
        header = [
            "House",
            *features,
            "Bias"
        ]
        file.write(",".join(header) + "\n")


def save_theta_in_file(
        hogwarts_house: str,
        theta: NDArray[np.float64],
        theta_file: Path
) -> None:
    """Ajoute au fichier de poids la ligne d'une maison."""

    with theta_file.open("a") as f:
        values: list[str] = [
            hogwarts_house, *(str(value) for value in theta)
        ]
        f.write(",".join(values) + "\n")


def initialize_result_file(
        result_file: Path
) -> None:
    """Crée le fichier de prédictions avec sa ligne d'en-tête."""

    with result_file.open("w") as file:
        header = [
            "Index",
            "Hogwarts House"
        ]
        file.write(",".join(header) + "\n")


def save_result_in_file(
        index: int,
        hogwarts_house: str,
        result_file: Path
) -> None:
    """Ajoute au fichier de prédictions la ligne d'un élève."""

    with result_file.open("a") as f:
        values: list[str] = [
            str(index),
            hogwarts_house
        ]
        f.write(",".join(values) + "\n")
