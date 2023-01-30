"""Contains definition of graph and it's nodes.

Computational graph is a form of directed acyclic graph that represents a
mathematical expression. In other words - tracker of simple operations. Thanks
to this graph, it becomes possible to apply Automatic Differentiation(AD).

AD - a technique to calculate the derivative of a function given by
an algorithm. AD takes advantage of the fact that an arbitrary function in
a computer program will still be calculated using arithmetic operations
(+, -, *, /) and elementary functions of standard libraries
(exp, log, sin, etc.). By applying the chain rule, the derivative of any order
can be calculated for a number of operations that is proportional to the number
of operations to calculate the function itself.

Each graph consists of nodes. Nodes are divided into:
    - Variable - a basic node with some changeable value
    - Constant - a node with a fixed value
    - Placeholder - a node with no value, so the value can be set later
    - Operation - a node that performs computations
"""
import numbers
import itertools
import numpy as np

from collections import deque
from collections.abc import Iterable


__all__ = (
    'Node', 'Constant', 'Variable', 'Placeholder', 'Operation', 'Graph',
    'get_current_graph', 'reset_current_graph', 'topological_sort',
    'add', 'mul', 'div', 'pow', 'dot', 'max', 'min', 'sum', 'mean',
    'sqrt', 'abs', 'exp', 'log', 'log2', 'log10', 'sin', 'cos', 'node_wrapper'
)


# disabled W0603(global-statement) until stack will be implemented
# pylint: disable=W0603
# TODO: graph thread-safe stack to keep multiple graphs
_GRAPH = None  # var to store current graph


def get_current_graph():
    """Return current graph. If it is `None` than create a new one."""
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = Graph()

    return _GRAPH


def reset_current_graph():
    """Set current graph to None"""
    global _GRAPH
    _GRAPH = None


class Graph:
    """Computational graph class."""

    count = itertools.count().__next__

    def __init__(self):
        """Constructor method
        """
        self.name = f'graph-{Graph.count()}'

    def as_default(self):
        """Set global graph to self."""
        global _GRAPH
        _GRAPH = self

    def __enter__(self):
        self.as_default()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.count = itertools.count().__next__
        reset_current_graph()


class NodeMixin:
    """Contains different useful members for nodes."""
    graph = lambda self: get_current_graph()
    count = itertools.count().__next__


# E1101(no-member) is disabled because this class should only
# be used to detect Node objects
# pylint: disable=E1101
class Node(NodeMixin):
    """Base node class."""


class Constant(Node):
    """Represents a node with a fixed value.

    :param value: value to set
    :param name: name of the node, if none than name will be
        created automatically, defaults to None
    :param dtype: value type, defaults to None
    :raises ValueError: when trying to change the value
    """

    def __init__(self, value, name=None, dtype=None):
        """Constructor method.
        """
        self._value = np.asarray(value, dtype=dtype)
        self.name = name or f"{self.graph().name}/constant-{self.count()}"
        self.gradient = None

    @property
    def value(self):
        """Get value of a node."""
        return self._value

    @value.setter
    def value(self, value):
        """Set value of a node.

        Constant value cannot be changed, so ValueError will be
        raised when trying.
        """
        raise ValueError("Cannot reassign constant.")

    def __str__(self):
        return self.name


class Placeholder(Node):
    """Represents a node with no value, so the value can be set later.

    .. note::
        The node name is crucial here, it can be used later by
        :class:`Session` to fill this node with a value.

        >>> x = Placeholder('a')
        >>> Session().run(x, feed_dict={'x': 1}) # will raise KeyError
        >>> Session().run(x, feed_dict={'a': 1}) # is OK

        In particular, you can fill it manually, but before computing
        the graph output.

        >>> x = Placeholder('a')
        >>> x.value = 1
        >>> Session().run(x, feed_dict={'x': 1}) # is OK

    :param name: node name, if name is None than it will be
        created automatically
    :raises ValueError: if Placeholder is initialized with name None
    """

    def __init__(self, name):
        """Constructor method.
        """
        self._value = None
        self.gradient = None

        if name is None:
            raise ValueError('Placeholder name cannot be None.')
        self.name = name

    @property
    def value(self):
        """Get value of a node."""
        return self._value

    @value.setter
    def value(self, value):
        """Set value of a node."""
        self._value = np.asarray(value)

    def __str__(self):
        return self.name


class Variable(Node):
    """Represents a basic node with some changeable value.

    :param value: value to set
    :param name: name of the node, if none than name will be
        created automatically, defaults to None
    :param dtype: value type
    """

    def __init__(self, value, name=None, dtype=None):
        self.value = np.asarray(value, dtype=dtype)
        self.gradient = None
        self.name = name or f"{self.graph().name}/variable-{self.count()}"

    def __str__(self):
        return self.name


class Operation(Node):
    """Base operation class. Represents a node that performs computations.

    Basically, all operations are divided by the number of operands into
    unary (1 operand), binary (2 operands) and default (depends on
    implementation). This is a base class, so it cannot be used for
    graph computations.

    Note: all operation nodes have similar name:
    graph-#/operator-`Operator Name`-#
    the only thing that can be specified explicitly is `Operator Name`

    :param op_name: operator name, defaults to 'Operator'
    :type op_name: str
    """

    def __init__(self, op_name='Operator'):
        """Constructor method.
        """
        self.inputs = ()
        self.name = f'{self.graph().name}/operator-{op_name}-{self.count()}'
        self.gradient = None
        self.value = None

    def forward(self, *args, **kwargs):
        """Return output of the operation by given inputs."""
        raise NotImplementedError("Must be implemented in child classes.")

    def backward(self, *args, **kwargs):
        """Return gradient of the operation by given inputs."""
        raise NotImplementedError("Must be implemented in child classes.")

    def __str__(self):
        return self.name


class UnaryOperator(Operation):
    """Operation subclass that defines the base class for unary operators."""

    def forward(self, value):
        """Return output of the operation by given input."""
        raise NotImplementedError("Must be implemented in child classes.")

    def backward(self, value, dout):
        """Return gradient of the operation by given input."""
        raise NotImplementedError("Must be implemented in child classes.")


class BinaryOperator(Operation):
    """Operation subclass that defines the base class for binary operations."""

    def forward(self, left, right):
        """Return output of the operation by given inputs."""
        raise NotImplementedError("Must be implemented in child classes.")

    def backward(self, left, right, dout):
        """Return gradient of the operation by given inputs."""
        raise NotImplementedError("Must be implemented in child classes.")


class Sum(Operation):
    """Sum of array elements over a given axis.

    :param value: array to sum
    :param axis: axis along which to sum, if None - sum all elements of array,
        defaults to None
    :param name: node name, defaults to 'sum'
    """
    def __init__(self, value, axis=None, name='sum'):
        """Constructor method
        """
        super().__init__(name)
        self.inputs = value,
        self.axis = axis

    def forward(self, value):
        """Return output of the operation by given input.

        :param value: input
        :return: sum of array over a given axis
        """
        return np.sum(value, axis=self.axis)

    def backward(self, value, dout):
        """Return gradient of the operation by given input.

        :param value: input
        :param dout: gradient of the path to this node
        :return: gradient of the operation
        """
        # implementation details: any simple sum gradient is a vector
        # or matrix of ones. So, it is enough to reshape gradient of
        # the path to this node into shape of this Operation result
        # and repeat it along Operation axis
        value = np.asarray(value)

        output_shape = np.array(value.shape)
        output_shape[self.axis] = 1

        tile_scaling = value.shape // output_shape
        dout = np.reshape(dout, output_shape)

        return np.tile(dout, tile_scaling)


class Mean(Operation):
    """Mean value of array over a given axis.

    :param value: array to get the mean from
    :param axis: axis along which to get mean, if None - get mean
        of the whole array, defaults to None
    :param name: node name, defaults to 'mean'
    """
    def __init__(self, value, axis=None, name='mean'):
        """Constructor method
        """
        super().__init__(name)
        self.inputs = value,
        self.axis = axis

    def forward(self, value):
        """Return output of the operation by given input.

        :param value: input
        :return: mean of array over a given axis
        """
        return np.mean(value, axis=self.axis)

    def backward(self, value, dout):
        """Return gradient of the operation by given input.

        :param value: input
        :param dout: gradient of the path to this node
        :return: gradient of the operation
        """
        # see the `Sum` backward implementation
        # the only difference between these two gradients is a constant,
        # but it doesn't change anything - d(c*x) = c*dx
        value = np.asarray(value)

        output_shape = np.array(value.shape)
        output_shape[self.axis] = 1

        tile_scaling = value.shape // output_shape
        dout = np.reshape(dout, output_shape)

        return np.tile(dout, tile_scaling) / value.size


class Add(BinaryOperator):
    """Element-wise sum.

    :param left: left operand of the operation
    :param right: right operand of the operation
    :param name: node name, defaults to 'add'
    """
    def __init__(self, left, right, name='add'):
        """Constructor method
        """
        super().__init__(name)
        self.inputs = (left, right)

    def forward(self, left, right):
        """Return output of the operation by given input.

        :param left: left operand
        :param right: right operand
        :return: element-wise sum
        """
        return np.add(left, right)

    def backward(self, left, right, dout):
        """Return gradient of the operation by given input.

        :param left: left operand input
        :param right: right operand input
        :param dout: gradient of the path to this node
        :return: gradient of the operation w.r.t both operands
        """
        def _get_sum_gradient(inp, dout_wrt_inp):
            """Get `add` gradient in the case of different operands shapes."""
            while np.ndim(dout_wrt_inp) > len(inp.shape):
                dout_wrt_inp = np.sum(dout_wrt_inp, axis=0)

            for axis, size in enumerate(inp.shape):
                if size == 1:
                    dout_wrt_inp = np.sum(
                        dout_wrt_inp,
                        axis=axis,
                        keepdims=True
                    )

            return dout_wrt_inp

        left, right = np.asarray(left), np.asarray(right)
        dout_wrt_left = _get_sum_gradient(left, dout)
        dout_wrt_right = _get_sum_gradient(right, dout)
        return dout_wrt_left, dout_wrt_right


class Multiply(BinaryOperator):
    """Element-wise multiply.

    :param left: left operand of the operation
    :param right: right operand of the operation
    :param name: node name, defaults to 'multiply'
    """
    def __init__(self, left, right, name='multiply'):
        """Constructor method
        """
        super().__init__(name)
        self.inputs = (left, right)

    def forward(self, left, right):
        """Return output of the operation by given input.

        :param left: left operand
        :param right: right operand
        :return: element-wise multiply
        """
        return np.multiply(left, right)

    def backward(self, left, right, dout):
        """Return gradient of the operation by given input.

        :param left: left operand input
        :param right: right operand input
        :param dout: gradient of the path to this node
        :return: gradient of the operation w.r.t both operands
        """
        return np.multiply(dout, right), np.multiply(dout, left)


class Divide(BinaryOperator):
    """Element-wise divide.

    :param left: left operand of the operation
    :param right: right operand of the operation
    :param name: node name, defaults to 'divide'
    """
    def __init__(self, left, right, name='divide'):
        super().__init__(name)
        self.inputs = (left, right)

    def forward(self, left, right):
        """Return output of the operation by given input.

        :param left: left operand
        :param right: right operand
        :return: element-wise divide
        """
        return np.divide(left, right)

    def backward(self, left, right, dout):
        """Return gradient of the operation by given input.

        :param left: left operand input
        :param right: right operand input
        :param dout: gradient of the path to this node
        :return: gradient of the operation w.r.t both operands
        """
        left, right = np.asarray(left), np.asarray(right)
        return dout / right, np.negative(dout) * left / np.power(right, 2)


class Power(BinaryOperator):
    """Power operator.



    :param left: left operand of the operation
    :param right: right operand of the operation
    :param name: node name, defaults to 'power'
    """
    def __init__(self, left, right, name='power'):
        """Constructor method
        """
        super().__init__(name)
        self.inputs = (left, right)

    def forward(self, left, right):
        """Return output of the operation by given input.

        :param left: base
        :param right: power
        :raises ValueError: integers to negative integer powers are not
         allowed (specific behavior of numpy)
        :return: left to the power of right
        """
        return np.power(left, right)

    def backward(self, left, right, dout):
        """Return gradient of the operation by given input.

        :param left: base
        :param right: power
        :param dout: gradient of the path to this node
        :return: gradient of the operation w.r.t both operands
        """
        left, right = np.asarray(left), np.asarray(right)
        return np.multiply(dout, right * np.power(left, right - 1)),\
            np.multiply(dout, np.log(left) * np.power(left, right))


class Matmul(BinaryOperator):
    """Matrix multiplication.

    :param left: left operand of the operation
    :param right: right operand of the operation
    :param name: node name, defaults to 'matmul'
    """
    def __init__(self, left, right, name='matmul'):
        """Constructor method
        """
        super().__init__(name)
        self.inputs = (left, right)

    def forward(self, left, right):
        """Return output of the operation by given input.

        :param left: left operand
        :param right: right operand
        :return: dot product of two arrays
        """
        return np.dot(left, right)

    def backward(self, left, right, dout):
        """Return gradient of the operation by given input.

        :param left: left operand
        :param right: right operand
        :param dout: gradient of the path to this node
        :return: gradient of the operation w.r.t both operands
        """
        left, right = np.asarray(left), np.asarray(right)
        return np.dot(dout, right.T), np.dot(left.T, dout)


class Max(BinaryOperator):
    """Element-wise maximum.

    :param left: left operand of the operation
    :param right: right operand of the operation
    :param name: node name, defaults to 'max'
    """
    def __init__(self, left, right, name='max'):
        """Constructor method
        """
        super().__init__(name)
        self.inputs = (left, right)

    def forward(self, left, right):
        """Return output of the operation by given input.

        :param left: left operand
        :param right: right operand
        :return: element-wise maximum of array elements
        """
        return np.maximum(left, right)

    def backward(self, left, right, dout):
        """Return gradient of the operation by given input.

        :param left: left operand
        :param right: right operand
        :param dout: gradient of the path to this node
        :return: gradient of the operation w.r.t both operands
        """
        return dout * np.where(left > right, 1, 0), \
            dout * np.where(left > right, 0, 1)


class Min(BinaryOperator):
    """Element-wise maximum.

    :param left: left operand of the operation
    :param right: right operand of the operation
    :param name: node name, defaults to 'max'
    """
    def __init__(self, left, right, name='min'):
        """Constructor method
        """
        super().__init__(name)
        self.inputs = (left, right)

    def forward(self, left, right):
        """Return output of the operation by given input.

        :param left: left operand
        :param right: right operand
        :return: element-wise minimum of array elements
        """
        return np.minimum(left, right)

    def backward(self, left, right, dout):
        """Return gradient of the operation by given input.

        :param left: left operand
        :param right: right operand
        :param dout: gradient of the path to this node
        :return: gradient of the operation w.r.t both operands
        """
        return dout * np.where(left < right, 1, 0), \
            dout * np.where(left < right, 0, 1)


class Sqrt(UnaryOperator):
    """Element-wise square root.

    :param value: value to get square root of
    :param name: node name, defaults to 'sqrt'
    """
    def __init__(self, value, name='sqrt'):
        """Constructor method
        """
        super().__init__(name)
        self.inputs = value,

    def forward(self, value):
        """Return output of the operation by given input.

        :param value: input
        :return: square root of the value
        """
        return np.sqrt(value)

    def backward(self, value, dout):
        """Return gradient of the operation by given input.

        :param value: input
        :param dout: gradient of the path to this node
        :return: gradient of the operation
        """
        return dout / (2 * np.sqrt(value)),


class Abs(UnaryOperator):
    """Take the number absolute (element-wise for arrays).

    :param value: value to get square root of
    :param name: node name, defaults to 'sqrt'
    """
    def __init__(self, value, name='abs'):
        """Constructor method
        """
        super().__init__(name)
        self.inputs = value,

    def forward(self, value):
        """Return output of the operation by given input.

        :param value: input
        :return: square root of the value
        """
        return np.abs(value)

    def backward(self, value, dout):
        """Return gradient of the operation by given input.

        :param value: input
        :param dout: gradient of the path to this node
        :return: gradient of the operation
        """
        # implementation details: |x| can be written as sqrt(x**2),
        # so derivative of this function will be x/|x|
        return np.multiply(dout, (value / np.abs(value))),


class Exp(UnaryOperator):
    """Element-wise exponentiation.

    :param value: value to get exponent of
    :param name: node name, defaults to 'exp'
    """
    def __init__(self, value, name='exp'):
        """Constructor method
        """
        super().__init__(name)
        self.inputs = value,

    def forward(self, value):
        """Return output of the operation by given input.

        :param value: input
        :return: value exponent
        """
        return np.exp(value)

    def backward(self, value, dout):
        """Return gradient of the operation by given input.

        :param value: input
        :param dout: gradient of the path to this node
        :return: gradient of the operation
        """
        return np.multiply(dout, np.exp(value)),


class Log(UnaryOperator):
    """Element-wise natural logarithm.

    :param value: value to get natural logarithm of
    :param name: node name, defaults to 'log'
    """
    def __init__(self, value, name='log'):
        """Constructor method
        """
        super().__init__(name)
        self.inputs = value,

    def forward(self, value):
        """Return output of the operation by given input.

        :param value: input
        :return: natural logarithm of the value
        """
        return np.log(value)

    def backward(self, value, dout):
        """Return gradient of the operation by given input.

        :param value: input
        :param dout: gradient of the path to this node
        :return: gradient of the operation
        """
        return np.divide(dout, value),


class Log2(UnaryOperator):
    """Element-wise natural logarithm.

    :param value: value to get natural logarithm of
    :param name: node name, defaults to 'log'
    """
    def __init__(self, value, name='log2'):
        """Constructor method
        """
        super().__init__(name)
        self.inputs = value,

    def forward(self, value):
        """Return output of the operation by given input.

        :param value: input
        :return: natural logarithm of the value
        """
        return np.log2(value)

    def backward(self, value, dout):
        """Return gradient of the operation by given input.

        :param value: input
        :param dout: gradient of the path to this node
        :return: gradient of the operation
        """
        return np.divide(dout, np.multiply(value, np.log(2))),


class Log10(UnaryOperator):
    """Element-wise natural logarithm.

    :param value: value to get natural logarithm of
    :param name: node name, defaults to 'log'
    """
    def __init__(self, value, name='log10'):
        """Constructor method
        """
        super().__init__(name)
        self.inputs = value,

    def forward(self, value):
        """Return output of the operation by given input.

        :param value: input
        :return: natural logarithm of the value
        """
        return np.log10(value)

    def backward(self, value, dout):
        """Return gradient of the operation by given input.

        :param value: input
        :param dout: gradient of the path to this node
        :return: gradient of the operation
        """
        return np.divide(dout, np.multiply(value, np.log(10))),


class Sin(UnaryOperator):
    """Element-wise trigonometric sine

    :param value: value to get sin of
    :param name: node name, defaults to 'sin'
    """
    def __init__(self, value, name='sin'):
        super().__init__(name)
        self.inputs = value,

    def forward(self, value):
        """Return output of the operation by given input.

        :param value: input
        :return: sin of the value
        """
        return np.sin(value)

    def backward(self, value, dout):
        """Return gradient of the operation by given input.

        :param value: input
        :param dout: gradient of the path to this node
        :return: gradient of the operation
        """
        return np.multiply(dout, np.cos(value)),


class Cos(UnaryOperator):
    """Element-wise trigonometric cosine

    :param value: value to get cos of
    :param name: node name, defaults to 'cos'
    """
    def __init__(self, value, name='cos'):
        super().__init__(name)
        self.inputs = value,

    def forward(self, value):
        """Return output of the operation by given input.

        :param value: input
        :return: cos of the value
        """
        return np.cos(value)

    def backward(self, value, dout):
        """Return gradient of the operation by given input.

        :param value: input
        :param dout: gradient of the path to this node
        :return: gradient of the operation
        """
        return np.multiply(dout, (-np.sin(value))),


def topological_sort(head_node):
    """Perform topological sort for a given graph using DFS algorithm.

    :param head_node: node to start sorting from
    :param head_node: :class:`Node`
    :return: list  of sorted nodes
    :rtype: deque
    """
    visited = set()
    order = deque()

    def _dfs(node):
        """Depth-first search recursion helper."""
        nonlocal visited

        if node not in visited:
            visited.add(node)
            if isinstance(node, Operation):
                for input_node in node.inputs:
                    _dfs(input_node)

            order.append(node)

    _dfs(head_node)
    return order


def node_wrapper(func, *args, **kwargs):
    """Automatically convert numeric types to `Constant`.

    :raises TypeError: in case some operands are not Node
    """
    fnargs = []
    for arg in args:
        # in this implementation of the wrapper,
        # only numeric types are automatically converted
        # to a Constant node, which is important for tests
        if isinstance(arg, Node):
            fnargs.append(arg)
        else:
            try:
                fnargs.append(Constant(arg))
            except TypeError as e:
                raise TypeError(
                    f"Incompatible argument type: {type(arg)}."
                ) from e

    return func(*fnargs, **kwargs)


# disabled W0622 (redefined-builtin)
# max, min, pow, sum, etc. redefining built-ins for aesthetic purposes
# pylint: disable=W0622

# these functions should be imported
# into __init__.py, so they can replace similar ones from numpy
def add(this, other):
    """Add two operands."""
    return node_wrapper(Add, this, other)


def mul(this, other):
    """Multiply two operands."""
    return node_wrapper(Multiply, this, other)


def div(this, other):
    """Divide two operands."""
    return node_wrapper(Divide, this, other)


def pow(this, other):
    """Raise the first operand to the power of the second."""
    return node_wrapper(Power, this, other)


def dot(this, other):
    """Multiply two matrices."""
    return node_wrapper(Matmul, this, other)


def max(this, other):
    """Check if 'this' is greater than 'other'."""
    return node_wrapper(Max, this, other)


def min(this, other):
    """Check if 'this' is less than 'other'."""
    return node_wrapper(Min, this, other)


def sum(this, axis=None):
    """Sum of array elements over a given axis."""
    return node_wrapper(Sum, this, axis=axis)


def mean(this, axis=None):
    """Compute the arithmetic mean along the specified axis."""
    return node_wrapper(Mean, this, axis=axis)


def sqrt(this):
    """Return the square-root of an array(element-wise) or a number."""
    return node_wrapper(Sqrt, this)


def abs(this):
    """Return absolute value of an array(element-wise) or a number."""
    return node_wrapper(Abs, this)


def exp(this):
    """Calculate the exponential of an array(element-wise) or a number."""
    return node_wrapper(Exp, this)


def log(this):
    """Natural logarithm (element-wise for arrays)."""
    return node_wrapper(Log, this)


def log2(this):
    """Logarithm with base 2 (element-wise for arrays)."""
    return node_wrapper(Log2, this)


def log10(this):
    """Logarithm with base 10 (element-wise for arrays)."""
    return node_wrapper(Log10, this)


def sin(this):
    """Trigonometric sine (element-wise for arrays)."""
    return node_wrapper(Sin, this)


def cos(this):
    """Trigonometric cosine (element-wise for arrays)."""
    return node_wrapper(Cos, this)


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
Node.__invert__ = lambda this: node_wrapper(Add, Node.__neg__(this), -1)

Node.__sub__.__doc__ = "Subtract the first operand from the second."
Node.__rsub__.__doc__ = Node.__sub__.__doc__
Node.__rtruediv__.__doc__ = Node.__truediv__.__doc__
Node.__neg__.__doc__ = "Multiply operand by -1."
Node.__rpow__.__doc__ = Node.__pow__.__doc__
Node.__invert__.__doc__ = "Compute bit-wise inversion, or bit-wise NOT " \
                          "(works only for integers)."
