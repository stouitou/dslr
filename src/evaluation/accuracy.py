import pandas as pd
import sys
from src.helpers.data import (
    get_targets,
    load_csv_dataset,
    prepare_evaluation_data
)


def display_accuracy_score(
        dataset_truth: pd.DataFrame,
        dataset_prediction: pd.DataFrame,
        targets: list[str]
) -> None:

    y_true, y_predicted = prepare_evaluation_data(
        dataset_truth,
        dataset_prediction,
        targets
    )

    errors = 0
    for i in range(len(y_true)):
        if y_true[i] != y_predicted[i]:
            errors += 1

    accuracy_score = 100 - ((errors * 100) / len(y_true))

    print(accuracy_score)


def main(
        argc: int,
        argv: list[str]
) -> int:

    if argc != 3:
        print(f"Usage: {argv[0]} <dataset> <houses_file>.")
        return 1

    dataset_truth: pd.DataFrame | None = load_csv_dataset(argv[1])
    if dataset_truth is None:
        return 1

    dataset_prediction: pd.DataFrame | None = load_csv_dataset(argv[2])
    if dataset_prediction is None:
        return 1

    targets: list[str] = get_targets(dataset_truth)

    try:
        display_accuracy_score(
            dataset_truth,
            dataset_prediction,
            targets
        )

    except ValueError as error:
        print(f"Value error: {error}")
        return 1

    return 0


if __name__ == "__main__":

    sys.exit(main(len(sys.argv), sys.argv))
