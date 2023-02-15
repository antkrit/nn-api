"""Contains implementation of graph nodes for Automatic Differentiation(AD).

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
import itertools
import numpy as np
from collections import deque
from api.lib.autograd.graph import get_current_graph


__all__ = (
    'Node', 'Constant', 'Variable', 'Placeholder', 'Operation', 'node_wrapper',
    'topological_sort', 'add', 'mul', 'div', 'pow', 'dot', 'max', 'min',
    'sum', 'mean', 'sqrt', 'abs', 'exp', 'log', 'log2', 'log10', 'sin', 'cos',
)


class NodeMixin:
    """Contains different useful members for nodes."""
    count = itertools.count().__next__

    @staticmethod
    def current_graph():
        """Return current graph."""
        return get_current_graph()

    def prepare_graph(self, graph):
        graph.nodes.append(self)
        graph.head_node = self


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
        self.gradient = None

        graph = self.current_graph()
        self.name = name or f"{graph.name}/constant-{self.count()}"
        self.prepare_graph(graph)

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
        self.prepare_graph(self.current_graph())

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

        graph = self.current_graph()
        self.name = name or f"{graph.name}/variable-{self.count()}"
        self.prepare_graph(graph)

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

    :param name: operator name, defaults to 'Operator'
    :param threshold: some minute float value to avoid problems like div by 0
    """

    def __init__(self, name=None, threshold=0):
        """Constructor method.
        """
        self.inputs = ()
        self.gradient = None
        self.value = None
        self.threshold = threshold

        graph = self.current_graph()
        self.name = name or f'{graph.name}/operator-{self.count()}'
        self.prepare_graph(graph)

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
    :param threshold: some minute float value to avoid problems like div by 0,
        defaults to 0
    """
    def __init__(self, value, axis=None, name='sum', threshold=0):
        """Constructor method
        """
        super().__init__(name, threshold)
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
    :param threshold: some minute float value to avoid problems like div by 0,
        defaults to 0
    """
    def __init__(self, value, axis=None, name='mean', threshold=0):
        """Constructor method
        """
        super().__init__(name, threshold)
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
    :param threshold: some minute float value to avoid problems like div by 0,
        defaults to 0
    """
    def __init__(self, left, right, name='add', threshold=0):
        """Constructor method
        """
        super().__init__(name, threshold)
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
    :param threshold: some minute float value to avoid problems like div by 0,
        defaults to 0
    """
    def __init__(self, left, right, name='multiply', threshold=0):
        """Constructor method
        """
        super().__init__(name, threshold)
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
    :param threshold: some minute float value to avoid problems like div by 0,
        defaults to 10^-32
    """
    def __init__(self, left, right, name='divide', threshold=1e-32):
        super().__init__(name, threshold)
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
        d_wrt_right = left / (np.power(right, 2) + self.threshold)
        return dout / right, np.negative(dout) * d_wrt_right


class Power(BinaryOperator):
    """Power operator.

    :param left: left operand of the operation
    :param right: right operand of the operation
    :param name: node name, defaults to 'power'
    :param threshold: some minute float value to avoid problems like div by 0,
        defaults to 10^-32
    """
    def __init__(self, left, right, name='power', threshold=1e-32):
        """Constructor method
        """
        super().__init__(name, threshold)
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
        d_wrt_left = np.multiply(dout, right * np.power(left, right - 1))
        d_wrt_right = np.multiply(
            dout,
            np.log(left+self.threshold) * np.power(left, right)
        )
        return d_wrt_left, d_wrt_right


class Matmul(BinaryOperator):
    """Matrix multiplication.

    :param left: left operand of the operation
    :param right: right operand of the operation
    :param name: node name, defaults to 'matmul'
    :param threshold: some minute float value to avoid problems like div by 0,
        defaults to 0
    """
    def __init__(self, left, right, name='matmul', threshold=0):
        """Constructor method
        """
        super().__init__(name, threshold)
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
    :param threshold: some minute float value to avoid problems like div by 0,
        defaults to 0
    """
    def __init__(self, left, right, name='max', threshold=0):
        """Constructor method
        """
        super().__init__(name, threshold)
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
        d_wrt_left = dout * np.where(left > right, 1, 0)
        d_wrt_right = dout * np.where(left > right, 0, 1)
        return d_wrt_left, d_wrt_right


class Min(BinaryOperator):
    """Element-wise maximum.

    :param left: left operand of the operation
    :param right: right operand of the operation
    :param name: node name, defaults to 'max'
    :param threshold: some minute float value to avoid problems like div by 0,
        defaults to 0
    """
    def __init__(self, left, right, name='min', threshold=0):
        """Constructor method
        """
        super().__init__(name, threshold)
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
        d_wrt_left = dout * np.where(left < right, 1, 0)
        d_wrt_right = dout * np.where(left < right, 0, 1)
        return d_wrt_left, d_wrt_right


class Sqrt(UnaryOperator):
    """Element-wise square root.

    :param value: value to get square root of
    :param name: node name, defaults to 'sqrt'
    :param threshold: some minute float value to avoid problems like div by 0,
        defaults to 10^-32
    """
    def __init__(self, value, name='sqrt', threshold=1e-32):
        """Constructor method
        """
        super().__init__(name, threshold)
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
        return dout / (2 * np.sqrt(value)+self.threshold),


class Abs(UnaryOperator):
    """Take the number absolute (element-wise for arrays).

    :param value: value to get square root of
    :param name: node name, defaults to 'sqrt'
    :param threshold: some minute float value to avoid problems like div by 0,
        defaults to 10^-32
    """
    def __init__(self, value, name='abs', threshold=1e-32):
        """Constructor method
        """
        super().__init__(name, threshold)
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
        return np.multiply(dout, (value / (np.abs(value)+self.threshold))),


class Exp(UnaryOperator):
    """Element-wise exponentiation.

    :param value: value to get exponent of
    :param name: node name, defaults to 'exp'
    :param threshold: some minute float value to avoid problems like div by 0,
        defaults to 0
    """
    def __init__(self, value, name='exp', threshold=0):
        """Constructor method
        """
        super().__init__(name, threshold)
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
    :param threshold: some minute float value to avoid problems like div by 0,
        defaults to 10^-32
    """
    def __init__(self, value, name='log', threshold=1e-32):
        """Constructor method
        """
        super().__init__(name, threshold)
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
        return np.divide(dout, np.asarray(value)+self.threshold),


class Log2(UnaryOperator):
    """Element-wise natural logarithm.

    :param value: value to get natural logarithm of
    :param name: node name, defaults to 'log'
    :param threshold: some minute float value to avoid problems like div by 0,
        defaults to 10^-32
    """
    def __init__(self, value, name='log2', threshold=1e-32):
        """Constructor method
        """
        super().__init__(name, threshold)
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
        return np.divide(dout, np.multiply(value, np.log(2))+self.threshold),


class Log10(UnaryOperator):
    """Element-wise natural logarithm.

    :param value: value to get natural logarithm of
    :param name: node name, defaults to 'log'
    :param threshold: some minute float value to avoid problems like div by 0,
        defaults to 10^-32
    """
    def __init__(self, value, name='log10', threshold=1e-32):
        """Constructor method
        """
        super().__init__(name, threshold)
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
        return np.divide(dout, np.multiply(value, np.log(10))+self.threshold),


class Sin(UnaryOperator):
    """Element-wise trigonometric sine

    :param value: value to get sin of
    :param name: node name, defaults to 'sin'
    :param threshold: some minute float value to avoid problems like div by 0,
        defaults to 0
    """
    def __init__(self, value, name='sin', threshold=0):
        super().__init__(name, threshold)
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
    :param threshold: some minute float value to avoid problems like div by 0,
        defaults to 0
    """
    def __init__(self, value, name='cos', threshold=0):
        super().__init__(name, threshold)
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


def topological_sort(nodes):
    """Generates topological sort for a given graph using DFS algorithm.

    :param nodes: node to start sorting from
    :return: list  of sorted nodes
    """
    visited = set()
    order = []

    def _dfs(node):
        """Depth-first search recursion helper."""
        nonlocal visited

        if node not in visited:
            visited.add(node)
            if isinstance(node, Operation):
                for input_node in node.inputs:
                    _dfs(input_node)

            order.append(node)

    try:
        iterator = iter(nodes)
        for node in iterator:
            _dfs(node)
            yield order
    except TypeError:
        _dfs(nodes)
        yield order


def node_wrapper(node, *args, **kwargs):
    """Automatically convert non-Node types to `Constant`.

    :raises TypeError: in case some operands are not Node
    """
    fnargs = []
    for arg in args:
        # in this implementation of the wrapper,
        # only numeric types are automatically converted
        # to a Constant node
        if isinstance(arg, Node):
            fnargs.append(arg)
        else:
            try:
                fnargs.append(Constant(arg))
            except TypeError as e:
                raise TypeError(
                    f"Incompatible argument type: {type(arg)}."
                ) from e

    return node(*fnargs, **kwargs)


# disabled W0622 (redefined-builtin)
# max, min, pow, sum, etc. redefining built-ins for aesthetic purposes
# pylint: disable=W0622

def add(this, other, **kwargs):
    """Add two operands."""
    return node_wrapper(Add, this, other, **kwargs)


def mul(this, other, **kwargs):
    """Multiply two operands."""
    return node_wrapper(Multiply, this, other, **kwargs)


def div(this, other, **kwargs):
    """Divide two operands."""
    return node_wrapper(Divide, this, other, **kwargs)


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
