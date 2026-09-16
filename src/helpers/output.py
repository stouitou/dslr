import numpy as np
from numpy.typing import NDArray
from config import THETA_FILE


def initialize_theta_file(
        features: list[str]
    ) -> None:

    with THETA_FILE.open("w") as file:
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

    with THETA_FILE.open("a") as f:
        values: list[str] = [
            hogwarts_house, *(str(value) for value in theta)
        ]
        f.write(",".join(values) + "\n")
