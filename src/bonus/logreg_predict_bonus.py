import sys
from config import RESULT_MBGD_FILE, RESULT_SGD_FILE
from src.bonus.prediction_bonus import run_prediction


# L'option ne choisit que la destination : la prédiction elle-même est
# identique, seuls les poids fournis changent.
RESULT_FILES = {
    "--sgd": RESULT_SGD_FILE,
    "--mbgd": RESULT_MBGD_FILE
}


def main(
        argc: int,
        argv: list[str]
) -> int:
    """Prédit les maisons à partir de poids entraînés par un bonus."""

    if argc != 4 or argv[1] not in RESULT_FILES:
        print(f"Usage: {argv[0]} --sgd|--mbgd <dataset> <theta_file>.")
        return 1

    return run_prediction(argv[2], argv[3], RESULT_FILES[argv[1]])


if __name__ == "__main__":

    sys.exit(main(len(sys.argv), sys.argv))
