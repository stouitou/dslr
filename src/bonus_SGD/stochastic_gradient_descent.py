import numpy as np
from numpy.typing import NDArray
from config import SGD_LEARNING_RATE, SGD_N_EPOCHS
from src.helpers.logistic_regression import cross_enthropy_loss, gradient


def stochastic_gradient_descent(
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        theta: NDArray[np.float64],
        learning_rate: float = SGD_LEARNING_RATE,
        n_epochs: int = SGD_N_EPOCHS
) -> tuple[NDArray[np.float64], list[float]]:
    """Corrige theta après chaque élève, au lieu de moyenner tout le
    dataset comme le gradient_descent() du mandatory.
    """

    m = len(y)
    cost_history: list[float] = []

    for _ in range(n_epochs):

        # theta évolue pendant le passage : un ordre figé répéterait le
        # même biais à chaque epoch au lieu de le moyenner.
        order = np.random.permutation(m)

        for i in order:
            # X[i:i+1] et non X[i] : garde 2 dimensions, sinon le X.T
            # de gradient() ne transpose rien et le calcul est faux.
            theta -= learning_rate * gradient(
                X[i:i + 1],
                y[i:i + 1],
                theta
            )

        # Une mesure par epoch, sur tout le dataset : celle d'un seul
        # élève ne dirait rien, et 32 000 points seraient illisibles.
        cost_history.append(cross_enthropy_loss(X, y, theta))

    return theta, cost_history


def train_model_sgd(
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

    final_theta, cost_history = stochastic_gradient_descent(
        X,
        y,
        theta
    )

    return final_theta, cost_history
