"""Contains math operations definition."""
from api.core.autograd.node import (
    Abs,
    Add,
    Assign,
    AssignAdd,
    AssignDivide,
    AssignMultiply,
    Cos,
    Divide,
    Einsum,
    Exp,
    Flatten,
    Log,
    Log2,
    Log10,
    Matmul,
    Max,
    Mean,
    Min,
    Multiply,
    Node,
    Power,
    Reshape,
    Sin,
    Sqrt,
    Sum,
)
from api.core.autograd.utils import node_wrapper

__all__ = (
    "add",
    "mul",
    "div",
    "pow",
    "dot",
    "max",
    "min",
    "sin",
    "cos",
    "sum",
    "mean",
    "sqrt",
    "rsqrt",
    "abs",
    "exp",
    "log",
    "log2",
    "log10",
    "assign",
    "assign_add",
    "assign_mul",
    "assign_div",
    "einsum",
    "reshape",
    "flatten",
)


# disabled W0622 (redefined-builtin)
# max, min, pow, sum, etc. redefining built-ins for aesthetic purposes
# pylint: disable=W0622
def add(this, other, **kwargs):
    """Add two operands."""
    return node_wrapper(Add, this, other, **kwargs)


def assign_add(ref, operation, **kwargs):
    """Add operation with reference assignment."""
    return node_wrapper(AssignAdd, ref, operation, **kwargs)


def assign(ref, operation, **kwargs):
    """Assign operation to reference."""
    return node_wrapper(Assign, ref, operation, **kwargs)


def mul(this, other, **kwargs):
    """Multiply two operands."""
    return node_wrapper(Multiply, this, other, **kwargs)


def assign_mul(ref, operation, **kwargs):
    """Multiply two operands with reference assignment.."""
    return node_wrapper(AssignMultiply, ref, operation, **kwargs)


def div(this, other, **kwargs):
    """Divide two operands."""
    return node_wrapper(Divide, this, other, **kwargs)


def assign_div(ref, operation, **kwargs):
    """Divide two operands with reference assignment.."""
    return node_wrapper(AssignDivide, ref, operation, **kwargs)


def einsum(subscripts, *arrays, **kwargs):
    """Evaluates the Einstein summation convention on the operands."""
    delimiter = kwargs.pop("delimiter", "->")
    subscripts, o_subscript = _parse_subscripts(subscripts, delim=delimiter)

    return node_wrapper(
        Einsum,
        *arrays,
        subscripts=subscripts,
        o_subscript=o_subscript,
        delimiter=delimiter,
        **kwargs
    )


def _parse_subscripts(string, delim="->"):
    """Parse subscripts for einsum.

    :param string: subscripts to parse
    :param delim: str operator that separates input subscripts
        from output subscript, defaults to '->'
    :return: tuple, array of input subscripts and the output subscript
    """
    string = string.replace(",", "")
    inp, out = string.split(delim)
    return inp.split(), out


def reshape(this, to_shape, **kwargs):
    """Reshape array."""
    return node_wrapper(Reshape, this, to_shape=to_shape, **kwargs)


def flatten(this, **kwargs):
    """Flatten array.

    Returns 3d array. Ignore batch if this.ndim > 3,
    else add batch of size 1.
    """
    return node_wrapper(Flatten, this, **kwargs)


def pow(this, other, **kwargs):
    """Raise the first operand to the power of the second."""
    return node_wrapper(Power, this, other, **kwargs)


def dot(this, other, **kwargs):
    """Multiply two matrices."""
    return node_wrapper(Matmul, this, other, **kwargs)


def max(this, other, **kwargs):
    """Check if 'this' is greater than 'other'."""
    return node_wrapper(Max, this, other, **kwargs)


def min(this, other, **kwargs):
    """Check if 'this' is less than 'other'."""
    return node_wrapper(Min, this, other, **kwargs)


def sum(this, **kwargs):
    """Sum of array elements over a given axis."""
    return node_wrapper(Sum, this, **kwargs)


def mean(this, **kwargs):
    """Compute the arithmetic mean along the specified axis."""
    return node_wrapper(Mean, this, **kwargs)


def sqrt(this, **kwargs):
    """Return the square-root of an array(element-wise) or a number."""
    return node_wrapper(Sqrt, this, **kwargs)


def rsqrt(this, **kwargs):
    """Return reciprocal square root of an array(element-wise) or a number."""
    sqrt_op = node_wrapper(Sqrt, this)
    return node_wrapper(Divide, 1, sqrt_op, **kwargs)


def abs(this, **kwargs):
    """Return absolute value of an array(element-wise) or a number."""
    return node_wrapper(Abs, this, **kwargs)


def exp(this, **kwargs):
    """Calculate the exponential of an array(element-wise) or a number."""
    return node_wrapper(Exp, this, **kwargs)


def log(this, **kwargs):
    """Natural logarithm (element-wise for arrays)."""
    return node_wrapper(Log, this, **kwargs)


def log2(this, **kwargs):
    """Logarithm with base 2 (element-wise for arrays)."""
    return node_wrapper(Log2, this, **kwargs)


def log10(this, **kwargs):
    """Logarithm with base 10 (element-wise for arrays)."""
    return node_wrapper(Log10, this, **kwargs)


def sin(this, **kwargs):
    """Trigonometric sine (element-wise for arrays)."""
    return node_wrapper(Sin, this, **kwargs)


def cos(this, **kwargs):
    """Trigonometric cosine (element-wise for arrays)."""
    return node_wrapper(Cos, this, **kwargs)


Node.__add__ = add
Node.__mul__ = mul
Node.__truediv__ = div
Node.__pow__ = pow
Node.__matmul__ = dot

# Make sure that all lambda functions have the relevant docstring
Node.__radd__ = Node.__add__
Node.__rmul__ = Node.__mul__
Node.__sub__ = lambda this, other: node_wrapper(Add, this, -other)
Node.__rsub__ = lambda this, other: node_wrapper(Add, -this, other)
Node.__rtruediv__ = lambda this, other: node_wrapper(Divide, other, this)
Node.__neg__ = lambda this: node_wrapper(Multiply, this, -1)
Node.__rpow__ = lambda this, other: node_wrapper(Power, other, this)

Node.__sub__.__doc__ = "Subtract the first operand from the second."
Node.__rsub__.__doc__ = Node.__sub__.__doc__
Node.__rtruediv__.__doc__ = Node.__truediv__.__doc__
Node.__neg__.__doc__ = "Multiply operand by -1."
Node.__rpow__.__doc__ = Node.__pow__.__doc__
