"""Automatic differentiation root package.

Function differentiation is carried out automatically, using
tensorflow-like computational graphs. Computational graph is a
directed acyclic graph (DAG), which consists of Variables, Constants,
Placeholders and Operations. They are all used to track and remember
simple operations (add, multiply and so on) of complex functions
(mse, sigmoid, ...) so that Automatic Differentiation can be applied.

    >>> with Graph() as g:
    ...     x = Variable(0.5, name='x')
    ...     y = Variable(1, name='y')
    ...     out = 4*x + y
    ...     Session().run(out)  # forward pass 4 * 0.5 + 1 = 3.0
    ...     grads = gradients(out)  # backward pass
    ...     grads[x]  # d(out)/dx = 4.0
    ...     grads[y]  # d(out)/dy = 1.0
    3.0
    4.0
    1.0

Contains following modules:
    - `_util`: contains various useful classes and functions
    - `graph`: contains definition of graph and it's nodes, \
        math operators and operations
    - `math`: contain a `math` object with all traceable mathematical operations
    - `session`: contains classes and functions to work with graph
"""
from api.lib.autograd.math import math
from api.lib.autograd.session import Session, gradients, topological_sort
from api.lib.autograd.graph import (
    Graph, Node, Operation, Constant, Variable,
    Placeholder, BinaryOperator, UnaryOperator,
    get_current_graph
)
