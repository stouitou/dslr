import numpy as np
from numpy.typing import NDArray
from config import MBGD_BATCH_SIZE, MBGD_LEARNING_RATE, MBGD_N_EPOCHS
from src.helpers.logistic_regression import cross_enthropy_loss, gradient


def mini_batch_gradient_descent(
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        theta: NDArray[np.float64],
        learning_rate: float = MBGD_LEARNING_RATE,
        n_epochs: int = MBGD_N_EPOCHS,
        batch_size: int = MBGD_BATCH_SIZE
) -> tuple[NDArray[np.float64], list[float]]:
    """Corrige theta après chaque paquet d'élèves : le compromis entre le
    batch (tout le dataset) et le SGD (un seul élève).
    """

    m = len(y)
    cost_history: list[float] = []

    for _ in range(n_epochs):

        # Même raison qu'en SGD : theta évolue pendant le passage, un ordre
        # figé répéterait le même biais à chaque epoch.
        order = np.random.permutation(m)

        for start in range(0, m, batch_size):
            # Tableau d'indices : l'indexation conserve d'elle-même les
            # 2 dimensions, pas besoin de la tranche du SGD.
            batch = order[start:start + batch_size]
            theta -= learning_rate * gradient(
                X[batch],
                y[batch],
                theta
            )

        cost_history.append(cross_enthropy_loss(X, y, theta))

    return theta, cost_history


def train_model_mbgd(
        x: NDArray[np.float64],
        y: NDArray[np.float64],
        theta: NDArray[np.float64]
) -> tuple[NDArray[np.float64], list[float]]:
    """Équivalent bonus de train_model() : seule la descente change."""

    # Colonne de 1 : elle porte le biais theta_0.
    X = np.hstack((
        x,
        np.ones((x.shape[0], 1))
    ))

    final_theta, cost_history = mini_batch_gradient_descent(
        X,
        y,
        theta
    )

    return final_theta, cost_history
