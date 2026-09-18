import pandas as pd
import sys
from tabulate import tabulate
from src.helpers.data import load_csv_dataset
from src.helpers.statistics import get_statistics


def main(
        argc: int,
        argv: list[str]
) -> int:

    if argc != 2:
        print(f"Usage: {argv[0]} <dataset>.")
        return 1

    dataset: pd.DataFrame | None = load_csv_dataset(argv[1])
    if dataset is None:
        return 1

    statistics: dict[str, list[str | float]] | None = get_statistics(dataset)
    if statistics is None:
        return 1

    print(
        tabulate(
            statistics,
            headers="keys",
            floatfmt=".6f"
        )
    )

    return 0


if __name__ == "__main__":

    sys.exit(main(len(sys.argv), sys.argv))
