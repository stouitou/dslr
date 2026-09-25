import sys
from config import THETA_MBGD_FILE, THETA_SGD_FILE
from src.bonus.training_bonus import run_training
from src.bonus.stochastic_gradient_descent import train_model_sgd
from src.bonus.mini_batch_gradient_descent import train_model_mbgd


# Chaque option câble une descente et sa destination.
ALGORITHMS = {
    "--sgd": (train_model_sgd, THETA_SGD_FILE),
    "--mbgd": (train_model_mbgd, THETA_MBGD_FILE)
}


def main(
        argc: int,
        argv: list[str]
) -> int:
    """Entraîne le modèle avec l'un des deux optimiseurs bonus."""

    if argc != 3 or argv[1] not in ALGORITHMS:
        print(f"Usage: {argv[0]} --sgd|--mbgd <dataset>.")
        return 1

    train_model, theta_file = ALGORITHMS[argv[1]]

    return run_training(argv[2], train_model, theta_file)


if __name__ == "__main__":

    sys.exit(main(len(sys.argv), sys.argv))
