import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


def plot_dataset(
        x: NDArray[np.float64],
        y: NDArray[np.float64]
    ) -> None:

    plt.scatter(x, y)

    plt.title("Relationship Between Car Mileage and Price")
    plt.xlabel("Mileage")
    plt.ylabel("Price")

    plt.show()


def plot_cost_history(
        cost_history: list[float]
    ) -> None:

    plt.plot(cost_history)

    plt.title("Gradient descent convergence")
    plt.xlabel("Iterations")
    plt.ylabel("Mean squared error")

    plt.show()
