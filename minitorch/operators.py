"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


def mul(x: float, y: float) -> float:
    """Multiply two numbers.

    Args:
    ----
        x: A float number.
        y: A float number.

    Returns:
    -------
        The product of x and y.

    """
    return x * y


def id(x: float) -> float:
    """Return the number unchanged.

    Args:
    ----
        x: A float number.

    Returns:
    -------
        The number x.

    """
    return x


def add(x: float, y: float) -> float:
    """Add two numbers.

    Args:
    ----
        x: A float number.
        y: A float number.

    Returns:
    -------
        The sum of x and y.

    """
    return x + y


def neg(x: float) -> float:
    """Negate a number.

    Args:
    ----
        x: A float number.

    Returns:
    -------
        The negation of x.

    """
    return -x


def lt(x: float, y: float) -> float:
    """Check if the first number is less than the second number.

    Args:
    ----
        x: A float number.
        y: A float number.

    Returns:
    -------
        True if x is strictly less than y, otherwise False.

    """
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    """Check if the two numbers are equal.

    Args:
    ----
        x: A float number.
        y: A float number.

    Returns:
    -------
        True if x is equal to y, otherwise False.

    """
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    """Return the maximum of two numbers.

    Args:
    ----
        x: A float number.
        y: A float number.

    Returns:
    -------
        The maximum of x and y.

    """
    return x if x > y else y


def is_close(x: float, y: float) -> bool:
    """Check if the first number is close to the second number within a tolerance of 1e-2.

    Args:
    ----
        x: A float number.
        y: A float number.

    Returns:
    -------
        True if x is close to y within a tolerance of 1e-2, otherwise False.

    """
    return abs(x - y) < 1e-2


def sigmoid(x: float) -> float:
    """Calculate the sigmoid of a number.

    Args:
    ----
        x: A float number.

    Returns:
    -------
        The sigmoid of x.

    """
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Calculate the ReLU of a number.

    Args:
    ----
        x: A float number.

    Returns:
    -------
        The ReLU of x.

    """
    return x if x > 0 else 0.0


EPS = 1e-6


def log(x: float) -> float:
    """Calculate the natural logarithm of a number.

    Args:
    ----
        x: A float number.

    Returns:
    -------
        The natural logarithm of x.

    """
    return math.log(x + EPS)


def exp(x: float) -> float:
    """Calculate the exponential of a number.

    Args:
    ----
        x: A float number.

    Returns:
    -------
        The exponential of x.

    """
    return math.exp(x)


def inv(x: float) -> float:
    """Calculate the inverse of a number.

    Args:
    ----
        x: A float number.

    Returns:
    -------
        The inverse of x.

    """
    return 1.0 / x


def log_back(x: float, d: float) -> float:
    """Calculate the derivative of the natural logarithm multiplied by a second number.

    Args:
    ----
        x: A float number.
        d: A float number.

    Returns:
    -------
        Inverse of x multiplied by y.

    """
    return d / (x + EPS)


def inv_back(x: float, d: float) -> float:
    """Calculate the derivative of the inverse of a number multiplied by a second number.

    Args:
    ----
        x: A float number.
        d: A float number.

    Returns:
    -------
        The derivative of the inverse of x multiplied by y.

    """
    return (-1.0 / (x**2)) * d


def relu_back(x: float, d: float) -> float:
    """Calculate the derivative of the ReLU of a number multiplied by a second number.

    Args:
    ----
        x: A float number.
        d: A float number.

    Returns:
    -------
        The derivative of the ReLU of x multiplied by y.

    """
    return d if x > 0 else 0.0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """Higher order map.

    Args:
    ----
        fn: A function that takes a float and returns a float.

    Returns:
    -------
        A function that takes an iterable of floats, applies the function to each element, and returns an iterable of floats.

    """

    def _map(ls: Iterable[float]) -> Iterable[float]:
        ret = []
        for x in ls:
            ret.append(fn(x))
        return ret

    return _map


def zipWith(
    fn: Callable[[float, float], float],
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """Higher order zipWith.

    Args:
    ----
        fn: Combine two values.

    Returns:
    -------
        An iterable of floats.

    """

    def _zipWith(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
        ret = []
        for x, y in zip(ls1, ls2):
            ret.append(fn(x, y))
        return ret

    return _zipWith


def reduce(
    fn: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    """Higher order reduce.

    Args:
    ----
        fn: A function that takes two floats and returns a float.
        start: The starting value.

    Returns:
    -------
        A function that takes an iterable of floats and returns a float.

    """

    def _reduce(ls: Iterable[float]) -> float:
        ret = start
        for x in ls:
            ret = fn(ret, x)
        return ret

    return _reduce


def addLists(a: Iterable[float], b: Iterable[float]) -> Iterable[float]:
    """Add two lists together.

    Args:
    ----
        a: An iterable of floats.
        b: An iterable of floats.

    Returns:
    -------
        An iterable of floats.

    """
    return zipWith(add)(a, b)


def negList(a: Iterable[float]) -> Iterable[float]:
    """Negate a list.

    Args:
    ----
        a: An iterable of floats.

    Returns:
    -------
        An iterable of floats.

    """
    return map(neg)(a)


def sum(a: Iterable[float]) -> float:
    """Sum all the elements in a list.

    Args:
    ----
        a: An iterable of floats.

    Returns:
    -------
        A float.

    """
    return reduce(add, 0.0)(a)


def prod(a: Iterable[float]) -> float:
    """Take the product of lists.

    Args:
    ----
        a: An iterable of floats.

    Returns:
    -------
        A float.

    """
    return reduce(mul, 1.0)(a)
