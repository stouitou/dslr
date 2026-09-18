import sys
import pandas as pd
from src.helpers.data import load_csv_dataset
from src.helpers.plotting import (
    create_pair_plot,
    prepare_plot_data
)


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

    data, features, labels = prepare_plot_data(dataset)

    create_pair_plot(
        data,
        features,
        labels,
    )

    return 0


if __name__ == "__main__":

    sys.exit(main(len(sys.argv), sys.argv))
