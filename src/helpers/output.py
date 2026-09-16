import numpy as np
from numpy.typing import NDArray
from pathlib import Path
from config import THETA_FILE, RESULT_FILE


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


def get_theta_values(
        theta_file: Path,
        n: int
    ) -> dict[str, NDArray[np.float64]] | None:

    theta: dict[str, NDArray[np.float64]] = {}
    try:
        with theta_file.open() as f:
            f.readline()
            for line in f:
                array = line.strip().split(",")
                theta[array[0]] = np.array(array[1:], dtype=np.float64).reshape(n + 1, 1)

        return theta

    except OSError as error:
        print(error)
        return None


def initialize_result_file(
    ) -> None:

    with RESULT_FILE.open("w") as file:
        header = [
            "Index",
            "Hogwarts House"
        ]
        file.write(",".join(header) + "\n")


def save_result_in_file(
        index: int,
        hogwarts_house: str,
    ) -> None:

    with RESULT_FILE.open("a") as f:
        values: list[str] = [
            str(index),
            hogwarts_house
        ]
        f.write(",".join(values) + "\n")
