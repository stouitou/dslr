import numpy as np
from numpy.typing import NDArray


def sigmoid(
        z: NDArray[np.float64]
    ) -> NDArray[np.float64]:

    return 1 / (1 + np.e ** -z)


def model(
        X: NDArray[np.float64],
        theta: NDArray[np.float64]
    ) -> NDArray[np.float64]:

    return sigmoid((X).dot(theta))


def cross_enthropy_loss(
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        theta: NDArray[np.float64]
    ) -> float:

    m = len(y)
    predictions = model(X, theta)

    return -1 / m * np.sum(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))


def gradient(
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        theta: NDArray[np.float64]
    ) -> NDArray[np.float64]:

    m = len(y)

    return 1 / m * X.T.dot(model(X, theta) - y)


def gradient_descent(
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        theta: NDArray[np.float64],
        learning_rate: float = 0.01,
        n_iteration: int = 1000
    ) -> tuple[NDArray[np.float64], list[float]]:

    cost_history: list[float] = []

    for _ in range(n_iteration):
        theta -= learning_rate * gradient(X, y, theta)
        cost_history.append(cross_enthropy_loss(X, y, theta))
    
    return theta, cost_history
