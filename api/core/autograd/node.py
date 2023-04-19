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

from api.core.autograd.graph import get_current_graph

# this module is not logically separable
# pylint: disable=too-many-lines

# some classes have 5 or more arguments (which cannot be logically combined)
# due to a large number of parent classes
# e.g. `Add` -> `BinaryOperation` -> `Operation` -> `Node`
# pylint: disable=too-many-arguments

# disable numpy warnings due to some cases where a negative value
# is given to the input of the log function, causing a RuntimeWarning
# (although this does not affect the solution)
np.seterr(all="ignore")


__all__ = (
    "Node",
    "Variable",
    "Placeholder",
    "Constant",
    "Operation",
    "UnaryOperation",
    "BinaryOperation",
    "Abs",
    "Add",
    "Assign",
    "AssignAdd",
    "AssignDivide",
    "AssignMultiply",
    "Cos",
    "Divide",
    "Einsum",
    "Exp",
    "Log",
    "Log2",
    "Log10",
    "Matmul",
    "Max",
    "Mean",
    "Min",
    "Multiply",
    "Node",
    "Power",
    "Sin",
    "Sqrt",
    "Sum",
)


class NodeMixin:
    """Contains different useful members for nodes."""

    count = itertools.count().__next__

    @staticmethod
    def current_graph():
        """Return current graph."""
        return get_current_graph()


# E1101(no-member) is disabled because this class should only
# be used to detect Node objects (isinstance)
# pylint: disable=E1101
class Node(NodeMixin):
    """Base node class.

    :param value: value to set
    :param name: name of the node, if None than name will be
        created automatically, defaults to None
    :param shape: static shape of the data
    :raises ValueError: the data shape during initialization does not
        match the one set as an argument
    """

    def __init__(self, value, name, shape):
        """Constructor method."""
        self._value = value
        self.gradient = None
        self.shape = None

        self.graph = self.current_graph()
        self._prefix = name or "node"
        self.name = f"{self.graph.name}/{self._prefix}-{self.count()}"

        value_not_none = self._value is not None

        if value_not_none and not hasattr(self._value, "shape"):
            self._value = np.asarray(self._value)

        if shape is not None and not isinstance(shape, tuple):
            shape = tuple(shape)

        self.shape = self._value.shape if value_not_none else shape

    @property
    def value(self):
        """Get value of a node."""
        return self._value

    @value.setter
    def value(self, value):
        """Set value of a node.

        :param value: value to set
        :raises ValueError: if the user tries to set data, the form of which
            is different from the one set during placeholder initialization.
        """
        if value is not None:
            if not hasattr(value, "shape"):
                value = np.asarray(value, dtype=object)

            self._value = value
            self.shape = value.shape
        else:
            if self.shape:
                msg = f"Cannot match shapes {self.shape} and None"
                raise ValueError(msg)

            self._value = None
            self.shape = None


class Placeholder(Node):
    """Represents a node with no value, so the value can be set later.

    .. note::
        The node name is crucial here, it can be used later by
        :class:`Session` to fill this node with a value.

        .. code-block:: python

            from api.core.autograd import Session

            x = Placeholder('a')
            Session().run(x, feed_dict={'x': 1}) # will raise KeyError
            Session().run(x, feed_dict={'a': 1}) # is OK

        In particular, you can fill it manually, but before computing
        the graph output.

        .. code-block:: python

            from api.core.autograd import Session

            x = Placeholder('a')
            x.value = 1
            Session().run(x, feed_dict={'x': 1}) # is OK

    :raises ValueError: if Placeholder is initialized with name None
    """

    def __init__(self, name=None, shape=None):
        """Constructor method."""
        name = name or "placeholder"
        super().__init__(value=None, name=name, shape=shape)

    def __str__(self):
        return self.name


class Constant(Node):
    """Represents a node with a fixed value.

    :param value: value to set
    :param name: name of the node, if none than name will be
        created automatically, defaults to None
    :raises ValueError: when trying to change the value or the data shape
        during initialization does not match the one set as an argument
    """

    def __init__(self, value, name=None, shape=None):
        """Constructor method."""
        name = name or "constant"
        super().__init__(value=value, name=name, shape=shape)

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
        raise ValueError(f"Cannot reassign constant {self.name}.")

    def __str__(self):
        return self.name


class Variable(Node):
    """Represents a basic node with some changeable value."""

    def __init__(self, value, name=None, shape=None):
        name = name or "variable"
        super().__init__(value=value, name=name, shape=shape)

    @property
    def value(self):
        """Get value of a node."""
        return self._value

    @value.setter
    def value(self, value):
        """Set value of a node.

        :param value: value to set
        :raises ValueError: if the user tries to set data, the form of which
            is different from the one set during variable initialization.
        """
        if not hasattr(value, "shape"):
            value = np.asarray(value)

        self._value = value
        self.shape = value.shape

    def __str__(self):
        return self.name


class Operation(Node):
    """Base operation class. Represents a node that performs computations.

    Basically, all operations are divided by the number of operands into
    unary (1 operand), binary (2 operands) and default (depends on
    implementation). This is a base class, so it cannot be used for
    graph computations.

    :param threshold: small floating point value used to maintain numerical
        stability, defaults to 0
    """

    def __init__(self, name=None, shape=None, threshold=0):
        """Constructor method."""
        name = name or "operator"
        super().__init__(value=None, name=name, shape=shape)
        self.inputs = ()
        self.threshold = threshold

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
        if not isinstance(ref, Node):
            raise ValueError("Reference object must be of the Node class")

        self._ref = ref
        self.inputs = (self._ref, op)

        super().__init__(*args, **kwargs)

    @property
    def value(self):
        return self._ref.value

    @value.setter
    def value(self, value):
        self._ref.value = value

    def forward(self, ref, op):
        """Return output of the operation by given input."""
        raise NotImplementedError("Must be implemented in child classes.")

    def backward(self, value, dout):
        """Return gradient of the operation by given input."""
        raise NotImplementedError("Must be implemented in child classes.")


class Einsum(Operation):
    """Evaluates the Einstein summation convention on the operands.

    For details see:
    https://numpy.org/doc/stable/reference/generated/numpy.einsum.html

    :param arrays: arrays for the operation
    :param subscripts: array of labels of forms for summation
    :param o_subscript: label, output form
    :param delimiter: separates labels for summation and output form
    :param name: node name, defaults to 'mean'
    :param threshold: some minute float value to avoid problems like div by 0,
        defaults to 0
    """

    def __init__(
        self,
        *arrays,
        subscripts=None,
        o_subscript=None,
        delimiter="->",
        name="einsum",
        shape=None,
        threshold=0,
    ):
        """Constructor method."""
        super().__init__(name=name, shape=shape, threshold=threshold)
        self.inputs = arrays
        self._subscripts = subscripts
        self._o_subscript = o_subscript
        self._delimiter = delimiter

    def subscripts(self, reverse=None):
        """Return subscripts string.

        :param reverse: int, the index for which is needed
            to replace the value with the output form. This
            argument is used to implement the gradient.
        :return: subscripts string
        """
        subscripts = self._subscripts.copy()
        out_subscript = self._o_subscript

        if reverse is not None:
            out_subscript = subscripts[reverse]
            subscripts[reverse] = self._o_subscript

        return ", ".join(subscripts) + self._delimiter + out_subscript

    def forward(self, *values):
        """Return output of the operation by given input.

        :param values: input
        :return: sum of array over a given axis
        """
        return np.einsum(self.subscripts(), *values)

    def backward(self, *values, dout):
        """Return gradient of the operation by given input.

        :param values: input
        :param dout: gradient of the path to this node
        :return: gradient of the operation
        """
        values = list(values) or []

        def _subroutine(array, i):
            nonlocal dout
            _values = array.copy()
            _values[i] = dout

            return np.einsum(self.subscripts(reverse=i), *_values)

        result = [_subroutine(values, i) for i, _ in enumerate(values)]
        output = result[0] if len(result) == 1 else result

        return output


class Sum(UnaryOperation):
    """Sum of array elements over a given axis.

    :param value: array to sum
    :param axis: axis along which to sum, if None - sum all elements of array,
        defaults to None
    :param name: node name, defaults to 'sum'
    :param threshold: some minute float value to avoid problems like div by 0,
        defaults to 0
    """

    def __init__(self, value, axis=None, name="sum", shape=None, threshold=0):
        """Constructor method."""
        super().__init__(name=name, shape=shape, threshold=threshold)
        self.inputs = (value,)
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

    def __init__(self, value, axis=None, name="mean", shape=None, threshold=0):
        """Constructor method."""
        super().__init__(name=name, shape=shape, threshold=threshold)
        self.inputs = (value,)
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

        return np.divide(np.tile(dout, tile_scaling), value.size)


class Add(BinaryOperation):
    """Element-wise sum."""

    def __init__(self, left, right, name="add", shape=None, threshold=0):
        """Constructor method."""
        super().__init__(name=name, shape=shape, threshold=threshold)
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
            inp = np.asarray(inp)
            dout_wrt_inp = np.asarray(dout_wrt_inp)

            if not dout_wrt_inp.shape:
                return dout_wrt_inp

            while np.ndim(dout_wrt_inp) > len(inp.shape):
                dout_wrt_inp = np.sum(dout_wrt_inp, axis=0)

            for axis, size in enumerate(inp.shape):
                if size == 1:
                    dout_wrt_inp = np.sum(
                        dout_wrt_inp, axis=axis, keepdims=True
                    )

            return dout_wrt_inp

        dout_wrt_left = _get_sum_gradient(left, dout)
        dout_wrt_right = _get_sum_gradient(right, dout)
        return dout_wrt_left, dout_wrt_right


class AssignAdd(AssignOperation, Add):
    """Element-wise sum with value assignment.

    :param ref: left operand of the operation, result of the
        operation will be assigned to this node
    :param op: right operand of the operation
    :param name: node name, defaults to 'assign_add'
    """

    def __init__(self, ref, op, name="assign_add"):
        """Constructor method."""
        super().__init__(
            ref=ref,
            op=op,  # init AssignOperation
            left=ref,
            right=op,  # init Add
            name=name,
            threshold=0,
        )

    forward = Add.forward
    backward = Add.backward


class Assign(AssignOperation, Add):
    """Update ref by assigning value to it.

    Partial case of the addition operation (x = x* + 0).

    :param ref: left operand of the operation, result of the
        operation will be assigned to this node
    :param op: right operand of the operation
    :param name: node name, defaults to 'assign_add'
    """

    def __init__(self, ref, op, name="assign"):
        """Constructor method."""
        super().__init__(
            ref=ref,
            op=op,  # init AssignOperation
            left=op,
            right=Constant(0),  # init Add: ref = op + 0
            name=name,
            threshold=0,
        )

    forward = Add.forward
    backward = Add.backward


class Multiply(BinaryOperation):
    """Element-wise multiply."""

    def __init__(self, left, right, name="multiply", shape=None, threshold=0):
        """Constructor method."""
        super().__init__(name=name, shape=shape, threshold=threshold)
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

    def __init__(self, ref, op, name="assign_mul", threshold=0):
        """Constructor method."""
        super().__init__(
            ref=ref,
            op=op,  # init AssignOperation
            left=ref,
            right=op,  # init Multiply
            name=name,
            threshold=threshold,
        )

    forward = Multiply.forward
    backward = Multiply.backward


class Divide(BinaryOperation):
    """Element-wise divide."""

    def __init__(self, left, right, name="divide", shape=None, threshold=1e-32):
        """Constructor method."""
        super().__init__(name=name, shape=shape, threshold=threshold)
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

        _dout = np.negative(dout)
        d_wrt_right = np.divide(left, np.power(right, 2) + self.threshold)

        return np.divide(dout, right), np.multiply(_dout, d_wrt_right)


class AssignDivide(AssignOperation, Divide):
    """Element-wise divide with value assignment.

    :param ref: left operand of the operation, result of the
        operation will be assigned to this node
    :param op: right operand of the operation
    :param name: node name, defaults to 'assign_div'
    :param threshold: some minute float value to avoid problems like div by 0,
        defaults to 0
    """

    def __init__(self, ref, op, name="assign_div", threshold=0):
        """Constructor method."""
        super().__init__(
            ref=ref,
            op=op,  # init AssignOperation
            left=ref,
            right=op,  # init Divide
            name=name,
            threshold=threshold,
        )

    forward = Divide.forward
    backward = Divide.backward


class Power(BinaryOperation):
    """Power operator."""

    def __init__(self, left, right, name="power", shape=None, threshold=1e-32):
        """Constructor method."""
        super().__init__(name=name, shape=shape, threshold=threshold)
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
        d_wrt_left = np.multiply(
            dout, np.multiply(right, np.power(left, right - 1))
        )
        d_wrt_right = np.multiply(
            dout,
            np.multiply(np.log(left + self.threshold), np.power(left, right)),
        )
        return d_wrt_left, d_wrt_right


class Matmul(BinaryOperation):
    """Matrix multiplication."""

    def __init__(self, left, right, name="matmul", shape=None, threshold=0):
        """Constructor method."""
        super().__init__(name=name, shape=shape, threshold=threshold)
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
    """Element-wise maximum."""

    def __init__(self, left, right, name="max", shape=None, threshold=0):
        """Constructor method."""
        super().__init__(name=name, shape=shape, threshold=threshold)
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
    """Element-wise maximum."""

    def __init__(self, left, right, name="min", shape=None, threshold=0):
        """Constructor method."""
        super().__init__(name=name, shape=shape, threshold=threshold)
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
    """Element-wise square root."""

    def __init__(self, value, name="sqrt", shape=None, threshold=1e-32):
        """Constructor method."""
        super().__init__(name=name, shape=shape, threshold=threshold)
        self.inputs = (value,)

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
        return (np.divide(dout, (2 * np.sqrt(value) + self.threshold)),)


class Abs(UnaryOperation):
    """Take the number absolute (element-wise for arrays)."""

    def __init__(self, value, name="abs", shape=None, threshold=1e-32):
        """Constructor method."""
        super().__init__(name=name, shape=shape, threshold=threshold)
        self.inputs = (value,)

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
        abs_ = np.abs(value) + self.threshold
        return (np.multiply(dout, (np.divide(value, abs_))),)


class Exp(UnaryOperation):
    """Element-wise exponentiation."""

    def __init__(self, value, name="exp", shape=None, threshold=0):
        """Constructor method."""
        super().__init__(name=name, shape=shape, threshold=threshold)
        self.inputs = (value,)

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
        return (np.multiply(dout, np.exp(value)),)


class Log(UnaryOperation):
    """Element-wise natural logarithm."""

    def __init__(self, value, name="log", shape=None, threshold=1e-32):
        """Constructor method."""
        super().__init__(name=name, shape=shape, threshold=threshold)
        self.inputs = (value,)

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
        return (np.divide(dout, np.asarray(value) + self.threshold),)


class Log2(UnaryOperation):
    """Element-wise natural logarithm."""

    def __init__(self, value, name="log2", shape=None, threshold=1e-32):
        """Constructor method."""
        super().__init__(name=name, shape=shape, threshold=threshold)
        self.inputs = (value,)

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
        return (
            np.divide(dout, np.multiply(value, np.log(2)) + self.threshold),
        )


class Log10(UnaryOperation):
    """Element-wise natural logarithm."""

    def __init__(self, value, name="log10", shape=None, threshold=1e-32):
        """Constructor method."""
        super().__init__(name=name, shape=shape, threshold=threshold)
        self.inputs = (value,)

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
        return (
            np.divide(dout, np.multiply(value, np.log(10)) + self.threshold),
        )


class Sin(UnaryOperation):
    """Element-wise trigonometric sine"""

    def __init__(self, value, name="sin", shape=None, threshold=0):
        """Constructor method."""
        super().__init__(name=name, shape=shape, threshold=threshold)
        self.inputs = (value,)

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
        return (np.multiply(dout, np.cos(value)),)


class Cos(UnaryOperation):
    """Element-wise trigonometric cosine"""

    def __init__(self, value, name="cos", shape=None, threshold=0):
        """Constructor method."""
        super().__init__(name=name, shape=shape, threshold=threshold)
        self.inputs = (value,)

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
        return (np.multiply(dout, (-np.sin(value))),)
