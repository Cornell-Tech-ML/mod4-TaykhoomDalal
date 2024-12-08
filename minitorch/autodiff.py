from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Tuple, Protocol


# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
    -------
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    # TODO: Implement for Task 1.1.

    up_perturbed = vals[:arg] + (vals[arg] + epsilon,) + vals[arg + 1 :]
    down_perturbed = vals[:arg] + (vals[arg] - epsilon,) + vals[arg + 1 :]

    slope = (f(*up_perturbed) - f(*down_perturbed)) / (2 * epsilon)

    return slope


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        """Accumulates the derivative of the output with respect to this variable."""

    ...

    @property
    def unique_id(self) -> int:
        """Returns the unique identifier of this variable."""
        ...

    def is_leaf(self) -> bool:
        """Returns True if this variable is a leaf node."""
        ...

    def is_constant(self) -> bool:
        """Returns True if this variable is a constant."""
        ...

    @property
    def parents(self) -> Iterable["Variable"]:
        """Returns the parents of this variable."""
        ...

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Computes gradients of inputs using the chain rule."""
        ...


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
    ----
        variable: The right-most variable

    Returns:
    -------
        Non-constant Variables in topological order starting from the right.

    """
    # create a set to store visited variables
    visited = set()

    # create a list to store sorted variables
    sorted_vars = []

    def visit(node: Variable) -> None:
        """Visits the variable and its parents recursively."""
        if node.unique_id in visited or node.is_constant():
            return
        if not node.is_leaf():
            # visit all the parents of the variable
            for parent in node.parents:
                if not parent.is_constant():
                    visit(parent)
        # mark the variable as visited
        visited.add(node.unique_id)

        # once all the parents have been visited, add the variable to the sorted list
        sorted_vars.insert(0, node)

    visit(variable)

    return sorted_vars


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
    ----
        variable: The right-most variable
        deriv: Its derivative that we want to propagate backward to the leaves.

    Returns:
    -------
        No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.

    """
    # call topological sort
    sorted_vars = topological_sort(variable)

    # create dictionary to store variables and their derivatives
    derivatives = {}
    derivatives[variable.unique_id] = deriv

    # iterate through the sorted variables in backward order
    for var in sorted_vars:
        # if the variable is a leaf node, accumulate the derivative
        if var.is_leaf():
            # accumulate the derivative
            var.accumulate_derivative(derivatives[var.unique_id])

        # if the variable is not a leaf node
        else:
            # call .chain_rule on the last function in the history of the variable
            grads = var.chain_rule(derivatives[var.unique_id])

            # loop through all the Scalars+derivatives provided by the chain rule
            for parent, derivative in grads:
                if parent.is_constant():
                    continue
                # accumulate derivatives for the Scalar in the dictionary
                derivatives.setdefault(parent.unique_id, 0)
                derivatives[parent.unique_id] += derivative


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Returns the saved values."""
        return self.saved_values
