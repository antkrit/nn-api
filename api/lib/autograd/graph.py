"""Contains definition of graph and it's nodes.

Computational graph is a form of directed acyclic graph that represents a
mathematical expression. In other words - tracker of simple operations. Thanks
to this graph, it becomes possible to apply Automatic Differentiation(AD).

AD - a technique to calculate the derivative of a function given by an algorithm.
AD takes advantage of the fact that an arbitrary function in a computer program
will still be calculated using arithmetic operations (+, -, *, /) and elementary
functions of standard libraries (exp, log, sin, etc.). By applying the
chain rule, the derivative of any order can be calculated for a number of
operations that is proportional to the number of operations to calculate the
function itself.

Each graph consists of nodes. Nodes are divided into:
    - Variable - a basic node with some changeable value
    - Constant - a node with a fixed value
    - Placeholder - a node with no value, so the value can be set later
    - Operation - a node that performs computations
"""
# pylint: disable= R1707, W0603
import itertools
import numpy as np


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
        del self


_GRAPH = None  # var to store current graph


def get_current_graph():
    """Return current graph. If it is `None` than create a new one."""
    global _GRAPH
    if not _GRAPH:
        _GRAPH = Graph()

    return _GRAPH


# pylint: disable=E1101
class NodeMeta(type):
    """Node metaclass.

     Add the `_graph` attribute to node classes and
     update it for each instance.
     """
    def __call__(cls, *args, **kwargs):
        _g = get_current_graph()
        cls._graph = _g

        return super(NodeMeta, cls).__call__(*args, **kwargs)


class Node(metaclass=NodeMeta):
    """Base node class."""
    # count the number of nodes to use in instance names
    count = itertools.count().__next__


class Constant(Node):
    """Represents a node with a fixed value.

    :param value: value to set
    :param name: name of the node, if none than name will be
        created automatically, defaults to None
    :param dtype: value type
    :raises ValueError: when trying to change the value
    """

    def __init__(self, value, name=None, dtype=np.float64):
        """Constructor method
        """
        self._value = np.array(value, dtype=dtype)
        self.name = name or f"{self._graph.name}/constant-{self.count()}"
        self.gradient = None

    @property
    def value(self):
        """Get or set value of a node."""
        return self._value

    @value.setter
    def value(self, value):
        raise ValueError("Cannot reassign constant.")

    def __repr__(self):
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
    :type name: str
    """

    def __init__(self, name):
        """Constructor method
        """
        self._value = None
        self.gradient = None
        self.name = name or f"{self._graph.name}/placeholder-{self.count()}"

    @property
    def value(self):
        """Get or set value of a node."""
        return self._value

    @value.setter
    def value(self, value):
        self._value = np.array(value)

    def __repr__(self):
        return self.name


class Variable(Node):
    """Represents a basic node with some changeable value.

    :param value: value to set
    :param name: name of the node, if none than name will be
        created automatically, defaults to None
    :param dtype: value type
    """

    def __init__(self, value, name=None, dtype=np.float64):
        self.value = np.array(value, dtype=dtype)
        self.gradient = None
        self.name = name or f"{self._graph.name}/variable-{self.count()}"

    def __repr__(self):
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
        """Constructor method
        """
        self.inputs = ()
        self.name = f'{self._graph.name}/operator-{op_name}-{self.count()}'
        self.gradient = None
        self.value = None

    def forward(self, *args, **kwargs):
        """Return output of the operation by given inputs."""
        raise NotImplementedError("Must be implemented in child classes.")

    def backward(self, *args, **kwargs):
        """Return gradient of the operation by given inputs."""
        raise NotImplementedError("Must be implemented in child classes.")

    def __repr__(self):
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
        # implementation details: see the `Sum` backward implementation
        # the only difference between these two gradients is a constant,
        # but it doesn't change anything - d(c*x) = c*dx
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
        return left + right

    def backward(self, left, right, dout):
        """Return gradient of the operation by given input.

        :param left: left operand input
        :param right: right operand input
        :param dout: gradient of the path to this node
        :return: gradient of the operation w.r.t both operands
        """
        left, right = self.inputs[0].value, self.inputs[1].value
        dout_wrt_left, dout_wrt_right = dout, dout

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

        dout_wrt_left = _get_sum_gradient(left, dout_wrt_left)
        dout_wrt_right = _get_sum_gradient(right, dout_wrt_right)
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
        return left * right

    def backward(self, left, right, dout):
        """Return gradient of the operation by given input.

        :param left: left operand input
        :param right: right operand input
        :param dout: gradient of the path to this node
        :return: gradient of the operation w.r.t both operands
        """
        return dout * right, dout * left


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
        return left / right

    def backward(self, left, right, dout):
        """Return gradient of the operation by given input.

        :param left: left operand input
        :param right: right operand input
        :param dout: gradient of the path to this node
        :return: gradient of the operation w.r.t both operands
        """
        return dout / right, -dout * left / np.power(right, 2)


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
        return dout * right * np.power(left, (right - 1)),\
            dout * np.log(left) * np.power(left, right)


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
        return dout * np.where(left <= 0, 0, 1), \
            dout * np.where(right <= 0, 0, 1)


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
        return dout * np.exp(value),


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
        return dout / value,


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
        return dout * np.cos(value),


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
        return dout * (-np.sin(value)),


def _node_wrapper(func, this, other):
    """Automatically convert numeric types to `Constant`."""
    if not any(isinstance(arg, Node) for arg in (this, other)):
        raise TypeError("Incompatible types.")

    if isinstance(this, (int, float)):
        return func(Constant(this), other)
    if isinstance(other, (int, float)):
        return func(this, Constant(other))
    return func(this, other)


Node.__add__ = lambda this, other: _node_wrapper(Add, this, other)
Node.__sub__ = lambda this, other: _node_wrapper(Add, this, -other)
Node.__rsub__ = lambda this, other: _node_wrapper(Add, -this, other)
Node.__mul__ = lambda this, other: _node_wrapper(Multiply, this, other)
Node.__truediv__ = lambda this, other: _node_wrapper(Divide, this, other)
Node.__rtruediv__ = lambda this, other: _node_wrapper(Divide, other, this)
Node.__neg__ = lambda this: _node_wrapper(Multiply, this, Constant(-1))
Node.__pow__ = lambda this, other: _node_wrapper(Power, this, other)
Node.__rpow__ = lambda this, other: _node_wrapper(Power, other, this)
Node.__matmul__ = lambda this, other: _node_wrapper(Matmul, this, other)
Node.__radd__ = Node.__add__
Node.__rmul__ = Node.__mul__
Node.__div__ = Node.__truediv__
Node.max = lambda this, other: _node_wrapper(Max, this, other)
Node.sum = Sum
Node.mean = Mean
Node.sqrt = Sqrt
Node.exp = Exp
Node.log = Log
Node.sin = Sin
Node.cos = Cos
