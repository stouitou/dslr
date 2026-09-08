import sys
import pandas as pd
from tabulate import tabulate
from src.helpers.utils import load_csv_dataset
from src.helpers.statistics import (
    ft_max,
    ft_mean,
    ft_min,
    ft_percentile,
    ft_std
)


if __name__ == "__main__":

    try:
        if len(sys.argv) != 2:
            print(f"Usage: {sys.argv[0]} <dataset>.")
            sys.exit(1)

        dataset: pd.DataFrame = load_csv_dataset(sys.argv[1])
        if dataset is None:
            sys.exit(1)

        statistics = {
            "": ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
        }
        for column_name, column in dataset.items():
            if column.dtype == float:
                non_null_values = column.dropna()
                data = list(non_null_values)
                percentiles = ft_percentile(data)
                subject = column_name if len(column_name) < 15 else column_name[:12] + "..."
                statistics[subject] = [
                    len(non_null_values),
                    ft_mean(data),
                    ft_std(data),
                    ft_min(data),
                    percentiles[0],
                    percentiles[1],
                    percentiles[2],
                    ft_max(data)
                ]

        print(
            tabulate(
                statistics,
                headers="keys",
                floatfmt=".6f"
            )
        )

    except Exception as error:
        print("Program exited with a fatal error.", error)
