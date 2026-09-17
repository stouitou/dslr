import numpy as np
from numpy.typing import NDArray
from config import LEARNING_RATE, N_ITERATION


def initialize_theta(
        n: int
) -> NDArray[np.float64]:

    return np.zeros((
        n + 1,
        1
    ))


def destandardize_theta(
        standardized_theta: NDArray[np.float64],
        mean: NDArray[np.float64],
        std: NDArray[np.float64]
) -> NDArray[np.float64]:

    theta = standardized_theta.ravel()
    theta_features = theta[:-1]
    theta_bias = theta[-1]

    original_features: NDArray[np.float64] = theta_features / std
    original_bias: np.float64 = theta_bias - np.sum(
        theta_features * mean / std
    )

    return np.append(original_features, original_bias)


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

    return -1 / m * \
        np.sum(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))


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

    for _ in range(n_iteration):
        theta -= learning_rate * gradient(X, y, theta)

    return theta


def train_model(
        x: NDArray[np.float64],
        y: NDArray[np.float64],
        theta: NDArray[np.float64]
) -> tuple[NDArray[np.float64], list[float]]:

    X = np.hstack((
        x,
        np.ones((x.shape[0], 1))
    ))

    final_theta = gradient_descent(
        X,
        y,
        theta,
        learning_rate=LEARNING_RATE,
        n_iteration=N_ITERATION
    )

    return final_theta
