from typing import Any


def ft_mean(args: list[int | float]) -> float:
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


def ft_median(args: list[int | float]) -> float:
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


def ft_quartile(args: list[int | float]) -> list:
    """Calculate the quartile of a list of numbers."""

    if not all(isinstance(x, (int, float)) for x in args):
        raise TypeError("Non numeric value in list")

    if not args:
        print('ERROR')
        return None

    sort = sorted(args)
    n = len(sort)

    result = []
    result.append(float(sort[n // 4]))
    result.append(float(sort[(n * 3) // 4]))

    return result


def ft_var(args: list[int | float]) -> float:
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
    result /= len(args)

    return result


def ft_std(args: list[int | float]) -> float:
    """Calculate the standard deviation of a list of numbers."""

    if not all(isinstance(x, (int, float)) for x in args):
        raise TypeError("Non numeric value in list")

    if not args:
        print('ERROR')
        return None

    var = ft_var(args)
    # Standars deviation is the square root of variance.
    result = var ** 0.5

    return result


def ft_statistics(*args: Any, **kwargs: Any) -> None:
    """Calculate differents statistics."""

    try:
        for x in kwargs:
            match kwargs[x]:
                case 'mean':
                    result = ft_mean(list(args))
                case 'median':
                    result = ft_median(list(args))
                case 'quartile':
                    result = ft_quartile(list(args))
                case 'std':
                    result = ft_std(list(args))
                case 'var':
                    result = ft_var(list(args))
                case _:
                    result = None
            if result is not None:
                print(f'{kwargs[x]} : {result}')

    except TypeError as error:
        print('Type error:', error)
    except Exception as error:
        print('Error:', error)
