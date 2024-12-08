import math
import random
from dataclasses import dataclass
from typing import List, Tuple


def make_pts(N: int) -> List[Tuple[float, float]]:
    """Generate a list of N random 2D points.

    Args:
    ----
        N: The number of points to generate.

    Returns:
    -------
        A list of N points, each represented as a tuple of two floats.

    """
    X = []
    for i in range(N):
        x_1 = random.random()
        x_2 = random.random()
        X.append((x_1, x_2))
    return X


@dataclass
class Graph:
    N: int
    X: List[Tuple[float, float]]
    y: List[int]


def simple(N: int) -> Graph:
    """Generate a simple dataset where the label is 1 if the first coordinate is less than 0.5 and 0 otherwise.

    Args:
    ----
        N: The number of points to generate.

    Returns:
    -------
        A Graph object containing the generated data.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def diag(N: int) -> Graph:
    """Generate a dataset where the label is 1 if the sum of the coordinates is less than 0.5 and 0 otherwise.

    Args:
    ----
        N: The number of points to generate.

    Returns:
    -------
        A Graph object containing the generated data.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 + x_2 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def split(N: int) -> Graph:
    """Generate a dataset where the label is 1 if the first coordinate is less than 0.2 or greater than 0.8 and 0 otherwise.

    Args:
    ----
        N: The number of points to generate.

    Returns:
    -------
        A Graph object containing the generated data.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.2 or x_1 > 0.8 else 0
        y.append(y1)
    return Graph(N, X, y)


def xor(N: int) -> Graph:
    """Generate a dataset where the label is 1 if the first coordinate is less than 0.5 and the second coordinate is greater than 0.5 or vice versa and 0 otherwise.

    Args:
    ----
        N: The number of points to generate.

    Returns:
    -------
        A Graph object containing the generated data.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if ((x_1 < 0.5 and x_2 > 0.5) or (x_1 > 0.5 and x_2 < 0.5)) else 0
        y.append(y1)
    return Graph(N, X, y)


def circle(N: int) -> Graph:
    """Generate a dataset where the label is 1 if the point's distance from the origin (after shifting by 0.5) is greater than sqrt(0.1), and 0 otherwise.

    Args:
    ----
        N: The number of points to generate.

    Returns:
    -------
        A Graph object containing the generated data.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        x1, x2 = (x_1 - 0.5, x_2 - 0.5)
        y1 = 1 if x1 * x1 + x2 * x2 > 0.1 else 0
        y.append(y1)
    return Graph(N, X, y)


def spiral(N: int) -> Graph:
    """Generate a spiral dataset with N points where the first half of the points are in class 0 and the second half are in class 1.

    Args:
    ----
        N: The number of points to generate.

    Returns:
    -------
        A Graph object containing the generated data.

    """

    def x(t: float) -> float:
        """Calculate the x-coordinate of a point on a spiral."""
        return t * math.cos(t) / 20.0

    def y(t: float) -> float:
        """Calculate the y-coordinate of a point on a spiral."""
        return t * math.sin(t) / 20.0

    X = [
        (x(10.0 * (float(i) / (N // 2))) + 0.5, y(10.0 * (float(i) / (N // 2))) + 0.5)
        for i in range(5 + 0, 5 + N // 2)
    ]
    X = X + [
        (y(-10.0 * (float(i) / (N // 2))) + 0.5, x(-10.0 * (float(i) / (N // 2))) + 0.5)
        for i in range(5 + 0, 5 + N // 2)
    ]
    y2 = [0] * (N // 2) + [1] * (N // 2)
    return Graph(N, X, y2)


datasets = {
    "Simple": simple,
    "Diag": diag,
    "Split": split,
    "Xor": xor,
    "Circle": circle,
    "Spiral": spiral,
}
