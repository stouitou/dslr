import matplotlib.pyplot as plt


def plot_cost_history(
        cost_history: list[float]
    ) -> None:

    plt.plot(cost_history)

    plt.title("Gradient descent convergence")
    plt.xlabel("Iterations")
    plt.ylabel("Mean squared error")

    plt.show()
