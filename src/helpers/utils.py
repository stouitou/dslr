from pathlib import Path
import pandas as pd


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