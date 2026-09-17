import pandas as pd


def ft_min(
        args: list[int | float]
) -> float | None:
    """Calculate the minimum of a list of numbers."""

    if not all(isinstance(x, (int, float)) for x in args):
        raise TypeError("Non numeric value in list")

    if not args:
        print('ERROR')
        return None

    result = args[0]
    for x in args:
        if x < result:
            result = x

    return result


def ft_max(
        args: list[int | float]
) -> float | None:
    """Calculate the maximum of a list of numbers."""

    if not all(isinstance(x, (int, float)) for x in args):
        raise TypeError("Non numeric value in list")

    if not args:
        print('ERROR')
        return None

    result = args[0]
    for x in args:
        if x > result:
            result = x

    return result


def ft_mean(
        args: list[int | float]
) -> float | None:
    """Calculate the mean of a list of numbers."""

    if not all(isinstance(x, (int, float)) for x in args):
        raise TypeError("Non numeric value in list")

    if not args:
        print('ERROR')
        return None

    result = 0
    for x in range(len(args)):
        result += args[x]
    result /= len(args)

    return result


def ft_median(
        args: list[int | float]
) -> float | None:
    """Calculate the median of a list of numbers."""

    if not all(isinstance(x, (int, float)) for x in args):
        raise TypeError("Non numeric value in list")

    if not args:
        print('ERROR')
        return None

    sort = sorted(args)
    n = len(sort)
    mid = n // 2
    if n % 2 == 0:
        return (sort[mid - 1] + sort[mid]) / 2
    else:
        return sort[mid]


def ft_percentile(
        args: list[int | float]
) -> list | None:
    """Calculate the percentile of a list of numbers."""

    if not all(isinstance(x, (int, float)) for x in args):
        raise TypeError("Non numeric value in list")

    if not args:
        print('ERROR')
        return None

    sort = sorted(args)
    n = len(sort)
    result = []
    for p in [0.25, 0.5, 0.75]:
        position = (n - 1) * p
        index = int(position)
        fraction = position - index
        if fraction == 0:
            result.append(float(sort[index]))
        else:
            value = sort[index] + (sort[index + 1] - sort[index]) * fraction
            result.append(float(value))

    return result


def ft_var(
        args: list[int | float]
) -> float | None:
    """Calculate the variance of a list of numbers."""

    if not all(isinstance(x, (int, float)) for x in args):
        raise TypeError("Non numeric value in list")

    if not args:
        print('ERROR')
        return None

    if len(args) == 1:
        return 0.0

    mean = ft_mean(args)
    result = 0
    for x in range(len(args)):
        result += (args[x] - mean) ** 2
    result /= len(args) - 1

    return result


def ft_std(
        args: list[int | float]
) -> float | None:
    """Calculate the standard deviation of a list of numbers."""

    if not all(isinstance(x, (int, float)) for x in args):
        raise TypeError("Non numeric value in list")

    if not args:
        print('ERROR')
        return None

    var = ft_var(args)
    result = var ** 0.5

    return result


def ft_range(
        args: list[int | float]
) -> float | None:

    if not all(isinstance(x, (int, float)) for x in args):
        raise TypeError("Non numeric value in list")

    if not args:
        print('ERROR')
        return None

    minimum = ft_min(args)
    maximum = ft_max(args)

    return maximum - minimum


def get_statistics(
        dataset: pd.DataFrame
) -> dict[str, list[str | float]] | None:

    statistics: dict[str, list[str | float]] = {
        "": ["count", "mean", "var", "std", "min", "25%", "50%", "75%", "max", "rge"]
    }

    try:
        for column_name, column in dataset.items():
            if column.dtype != float:
                continue

            non_null_values = column.dropna()
            data = list(non_null_values)

            subject = (
                column_name
                if len(column_name) < 15
                else column_name[:12] + "..."
            )

            percentiles = ft_percentile(data)

            statistics[subject] = [
                len(non_null_values),
                ft_mean(data),
                ft_var(data),
                ft_std(data),
                ft_min(data),
                percentiles[0],
                percentiles[1],
                percentiles[2],
                ft_max(data),
                ft_range(data)
            ]

        return statistics

    except TypeError as error:
        print("Program exited with a fatal error:", error)
        return None
