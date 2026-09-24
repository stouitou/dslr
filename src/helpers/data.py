import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pathlib import Path
from config import RELEVANT_FEATURES
from src.helpers.output import get_theta_values


def load_csv_dataset(
        path: Path | str
) -> pd.DataFrame | None:

    try:
        path = Path(path)
        if not path.suffix == ".csv":
            raise ValueError("file must have .csv extension")

        df = pd.read_csv(path)
        print(f"Loading dataset of dimensions {df.shape}")

        return df

    except TypeError as error:
        print("Type error:", error)
    except ValueError as error:
        print("Value error:", error)
    except FileNotFoundError as error:
        print("File not found error:", error)
    except PermissionError as error:
        print("Permission error:", error)

    return None


def get_features(
        dataset: pd.DataFrame
) -> list[str]:

    features: list[str] = []

    for column_name, column in dataset.items():
        if column.dtype == float:
            features.append(column_name)

    return features


def get_targets(
        dataset: pd.DataFrame
) -> list[str]:

    labels: pd.Series = dataset["Hogwarts House"]
    targets: list[str] = list(np.unique(labels))

    return targets


def get_clean_data(
        dataset: pd.DataFrame,
        features: list[str]
) -> NDArray[np.float64]:

    data: pd.DataFrame = dataset[features].copy()
    clean_data: pd.DataFrame = data.fillna(data.mean())

    return clean_data.to_numpy()


def standardize_data(
        data: NDArray[np.float64]
) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64]
]:
    """Standardisation: z-score"""

    mean: NDArray[np.float64] = data.mean(axis=0)
    std: NDArray[np.float64] = data.std(axis=0)

    standardized_data: NDArray[np.float64] = (data - mean) / std

    return standardized_data, mean, std


def get_binary_target(
        dataset: pd.DataFrame,
        hogwarts_house: str
) -> NDArray[np.float64]:

    mask: pd.Series = dataset["Hogwarts House"] == hogwarts_house
    y: NDArray[np.float64] = mask.to_numpy(dtype=np.float64)
    y = y.reshape(y.shape[0], 1)

    return y


def prepare_prediction_data(
        dataset: pd.DataFrame,
        theta_file: Path
) -> tuple[
        NDArray[np.float64],
        dict[str, NDArray[np.float64]] | None
]:

    x: NDArray[np.float64] = get_clean_data(dataset, RELEVANT_FEATURES)
    X: NDArray[np.float64] = np.hstack((
        x,
        np.ones((x.shape[0], 1))
    ))
    theta: dict[str, NDArray[np.float64]] | None = get_theta_values(
        theta_file,
        len(RELEVANT_FEATURES)
    )

    return X, theta


def prepare_evaluation_data(
        dataset_truth: pd.DataFrame,
        dataset_prediction: pd.DataFrame,
        targets: list[str]
) -> tuple[NDArray[np.str_], NDArray[np.str_]]:

    target = "Hogwarts House"

    if target not in dataset_truth.columns:
        raise ValueError(
            f"'{target}' column is missing from truth dataset."
        )

    if dataset_truth[target].isna().any():
        raise ValueError(
            f"One or more values in the '{target}' column are missing."
        )

    if target not in dataset_prediction.columns:
        raise ValueError(
            f"'{target}' column is missing from prediction dataset."
        )

    truth = dataset_truth[target].to_numpy()
    prediction = dataset_prediction[target].to_numpy()

    if len(truth) != len(prediction):
        raise ValueError(
            "Truth and prediction datasets have different numbers of rows."
        )

    invalid_predictions = set(prediction) - set(targets)
    if invalid_predictions:
        raise ValueError(
            f"Unknown prediction classes: {invalid_predictions}"
        )

    return truth, prediction
