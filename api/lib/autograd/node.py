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
from api.lib.autograd.graph import get_current_graph


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
        """Constructor method."""
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


class Constant(Node):
    """Represents a node with a fixed value.

    :param value: value to set
    :param name: name of the node, if none than name will be
        created automatically, defaults to None
    :raises ValueError: when trying to change the value
    """

    def __init__(self, value, name=None):
        """Constructor method.
        """
        graph = self.current_graph()
        self.name = name or f"{graph.name}/constant-{self.count()}"
        self.prepare_graph(graph)

        self._value = value
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


class Variable(Node):
    """Represents a basic node with some changeable value.

    :param value: value to set
    :param name: name of the node, if none than name will be
        created automatically, defaults to None
    """

    def __init__(self, value, name=None):
        graph = self.current_graph()
        self.name = name or f"{graph.name}/variable-{self.count()}"
        self.prepare_graph(graph)

        self.value = value
        self.gradient = None

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
        """Constructor method."""
        self.inputs = ()
        self.threshold = threshold
        self.gradient = None

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


class UnaryOperation(Operation):
    """Operation subclass that defines the base class for unary operators."""

    def forward(self, value):
        """Return output of the operation by given input."""
        raise NotImplementedError("Must be implemented in child classes.")

    def backward(self, value, dout):
        """Return gradient of the operation by given input."""
        raise NotImplementedError("Must be implemented in child classes.")


class BinaryOperation(Operation):
    """Operation subclass that defines the base class for binary operations."""

    def forward(self, left, right):
        """Return output of the operation by given inputs."""
        raise NotImplementedError("Must be implemented in child classes.")

    def backward(self, left, right, dout):
        """Return gradient of the operation by given inputs."""
        raise NotImplementedError("Must be implemented in child classes.")


class AssignOperation(Operation):
    """Assign operations superclass.

    :param ref: left operand of the operation, result of the
    operation will be assigned to this node
    :param op: right operand of the operation
    """

    def __init__(self, ref, op, *args, **kwargs):
        """Constructor method."""
        self._ref = ref
        self.inputs = (self._ref, op)

        super().__init__(*args, **kwargs)

        if not isinstance(ref, Node):
            raise ValueError("Reference object must be of the Node class")

    @property
    def value(self):
        return self._ref.value

    @value.setter
    def value(self, value):
        if self._ref:
            self._ref.value = value

    def forward(self, ref, op):
        """Return output of the operation by given input."""
        raise NotImplementedError("Must be implemented in child classes.")

    def backward(self, value, dout):
        """Return gradient of the operation by given input."""
        raise NotImplementedError("Must be implemented in child classes.")


class Sum(UnaryOperation):
    """Sum of array elements over a given axis.

    :param value: array to sum
    :param axis: axis along which to sum, if None - sum all elements of array,
        defaults to None
    :param name: node name, defaults to 'sum'
    :param threshold: some minute float value to avoid problems like div by 0,
        defaults to 0
    """
    def __init__(self, value, axis=None, name='sum', threshold=0):
        """Constructor method."""
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


class Mean(UnaryOperation):
    """Mean value of array over a given axis.

    :param value: array to get the mean from
    :param axis: axis along which to get mean, if None - get mean
        of the whole array, defaults to None
    :param name: node name, defaults to 'mean'
    :param threshold: some minute float value to avoid problems like div by 0,
        defaults to 0
    """
    def __init__(self, value, axis=None, name='mean', threshold=0):
        """Constructor method."""
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


class Add(BinaryOperation):
    """Element-wise sum.

    :param left: left operand of the operation
    :param right: right operand of the operation
    :param name: node name, defaults to 'add'
    :param threshold: some minute float value to avoid problems like div by 0,
        defaults to 0
    """
    def __init__(self, left, right, name='add', threshold=0):
        """Constructor method."""
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


class AssignAdd(AssignOperation, Add):
    """Element-wise sum with value assignment.

    :param ref: left operand of the operation, result of the
        operation will be assigned to this node
    :param op: right operand of the operation
    :param name: node name, defaults to 'assign_add'
    :param threshold: some minute float value to avoid problems like div by 0,
        defaults to 0
    """

    def __init__(self, ref, op, name='assign_add', threshold=0):
        """Constructor method."""
        super().__init__(
            ref=ref, op=op,  # init AssignOperation
            left=ref, right=op,  # init Add
            name=name, threshold=threshold
        )

    forward = Add.forward
    backward = Add.backward


class Multiply(BinaryOperation):
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


class AssignMultiply(AssignOperation, Multiply):
    """Element-wise multiply with value assignment.

    :param ref: left operand of the operation, result of the
        operation will be assigned to this node
    :param op: right operand of the operation
    :param name: node name, defaults to 'assign_mul'
    :param threshold: some minute float value to avoid problems like div by 0,
        defaults to 0
    """

    def __init__(self, ref, op, name='assign_mul', threshold=0):
        """Constructor method."""
        super().__init__(
            ref=ref, op=op,  # init AssignOperation
            left=ref, right=op,  # init Multiply
            name=name, threshold=threshold
        )

    forward = Multiply.forward
    backward = Multiply.backward


class Divide(BinaryOperation):
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


class AssignDivide(AssignOperation, Divide):
    """Element-wise divide with value assignment.

    :param ref: left operand of the operation, result of the
        operation will be assigned to this node
    :param op: right operand of the operation
    :param name: node name, defaults to 'assign_div'
    :param threshold: some minute float value to avoid problems like div by 0,
        defaults to 0
    """

    def __init__(self, ref, op, name='assign_div', threshold=0):
        """Constructor method."""
        super().__init__(
            ref=ref, op=op,  # init AssignOperation
            left=ref, right=op,  # init Divide
            name=name, threshold=threshold
        )

    forward = Divide.forward
    backward = Divide.backward


class Power(BinaryOperation):
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


class Matmul(BinaryOperation):
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


class Max(BinaryOperation):
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


class Min(BinaryOperation):
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


class Sqrt(UnaryOperation):
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


class Abs(UnaryOperation):
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


class Exp(UnaryOperation):
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


class Log(UnaryOperation):
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


class Log2(UnaryOperation):
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


class Log10(UnaryOperation):
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


class Sin(UnaryOperation):
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


class Cos(UnaryOperation):
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
            order = []
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
