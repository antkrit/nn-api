from api.lib.utils import Container
from api.lib.autograd.node import (
    Node, Variable, Placeholder, Constant,
    Operation, UnaryOperation, BinaryOperation
)


nodes = Container(
    name='nodes',

    node=Node,
    constant=Constant,
    variable=Variable,
    placeholder=Placeholder,
    operation=Operation,
    unary_operation=UnaryOperation,
    binary_operation=BinaryOperation
)
