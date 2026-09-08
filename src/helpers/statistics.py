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
    # Standard deviation is the square root of variance.
    result = var ** 0.5

    return result
