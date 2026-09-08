import sys
import pandas as pd
from src.utils import load_csv_dataset


if __name__ == "__main__":

    try:
        if len(sys.argv) != 2:
            print(f"Usage: {sys.argv[0]} <dataset>.")
            sys.exit(1)

        dataset: pd.DataFrame = load_csv_dataset(sys.argv[1])
        if dataset is None:
            sys.exit(1)

        print(dataset.info())

        drop_null_dataset = dataset.dropna()
        print(drop_null_dataset.info())

        fill_null_dataset = dataset.fillna(0)
        print(fill_null_dataset.info())

        duplicate_removed_dataset = dataset.drop_duplicates()
        print(duplicate_removed_dataset.info())

        for feature in dataset.items():
            print(feature[0])
            print(len(feature[1]))
            # print(feature)

    except Exception as error:
        print("Program exited with a fatal error.", error)