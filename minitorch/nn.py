from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    # TODO: Implement for Task 4.3.

    new_height = height // kh
    new_width = width // kw

    # 1. reshape tensor
    reshaped = input.contiguous().view(batch, channel, new_height, kh, new_width, kw)

    # 2. permute tensor
    permuted = reshaped.permute(0, 1, 2, 4, 3, 5)

    # 3. flatten tensor and return
    return (
        permuted.contiguous().view(batch, channel, new_height, new_width, kh * kw),
        new_height,
        new_width,
    )


# TODO: Implement for Task 4.3.


def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Apply average pooling to input with kernel size.

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width

    """
    # 1. Tile the input tensor
    tiled, new_height, new_width = tile(input, kernel)

    # 2. Compute the mean of the last dimension
    pooled = tiled.mean(dim=4)

    # 3. remove the last dimension and return
    return pooled.view(input.shape[0], input.shape[1], new_height, new_width)


max_reduce = FastOps.reduce(operators.max, float("-inf"))


def argmax(input: Tensor, dim: int) -> Tensor:
    """Compute the argmax as a 1-hot tensor.

    Args:
    ----
        input : input tensor
        dim : dimension to apply argmax


    Returns:
    -------
        :class:`Tensor` : tensor with 1 on highest cell in dim, 0 otherwise

    """
    out = max_reduce(input, dim)
    return out == input


class Max(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: Tensor) -> Tensor:
        """Max function $f(x) = max(x)$"""
        max_dim = int(dim.item())
        ctx.save_for_backward(input, max_dim)
        return max_reduce(input, max_dim)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Backward of max should be argmax (see above)"""
        input, max_dim = ctx.saved_tensors

        # Compute the argmax
        return grad_output * argmax(input, max_dim), 0.0


def max(input: Tensor, dim: int) -> Tensor:
    """Compute the max of the input tensor over the given dimension.

    Args:
    ----
        input: tensor to reduce
        dim: dimension to reduce

    Returns:
    -------
        Tensor of size batch x channel x height x width

    """
    return Max.apply(input, input._ensure_tensor(dim))


def softmax(input: Tensor, dim: int) -> Tensor:
    """Compute the softmax of the input tensor over the given dimension.

    Args:
    ----
        input: tensor to apply softmax
        dim: dimension to apply softmax

    Returns:
    -------
        Tensor of size batch x channel x height x width

    """
    # use trick to avoid numerical instability
    max_input = max(input, dim)
    return (input - max_input).exp() / (input - max_input).exp().sum(dim)


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    """Compute the log of the softmax of the input tensor over the given dimension.

    Args:
    ----
        input: tensor to apply log softmax
        dim: dimension to apply log softmax

    Returns:
    -------
        Tensor of size batch x channel x height x width

    """
    # use logsumexp trick to avoid numerical instability
    max_input = max(input, dim)
    return input - max_input - (input - max_input).exp().sum(dim).log()


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Apply max pooling to input with kernel size.

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width

    """
    # 1. Tile the input tensor
    tiled, new_height, new_width = tile(input, kernel)

    # 2. Compute the max of the last dimension
    pooled = max(tiled, dim=4)

    # 3. remove the last dimension and return
    return pooled.view(input.shape[0], input.shape[1], new_height, new_width)


def dropout(input: Tensor, p: float, ignore: bool = False) -> Tensor:
    """Apply dropout to input tensor.

    Args:
    ----
        input: tensor to apply dropout
        p: probability of dropping out
        ignore: ignore the dropout (used for testing)

    Returns:
    -------
        Tensor of size batch x channel x height x width

    """
    if ignore:
        return input

    # verify that p is between 0 and 1
    assert 0.0 <= p <= 1.0

    return input * (rand(input.shape) > p)
