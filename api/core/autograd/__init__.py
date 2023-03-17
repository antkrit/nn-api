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
...     sess = Session()
...     sess.run(out)  # forward pass 4 * 0.5 + 1 = 3.0
...     grads = sess.gradients(out)  # backward pass
...     grads[x]  # d(out)/dx = 4.0
...     grads[y]  # d(out)/dy = 1.0
3.0
4.0
1.0

Contains following modules:
- `graph`: contains definition of graph
- `node`: contain definition of graph nodes, math operators and operations
- `session`: contains classes and functions to work with graph
"""
from api.core.autograd import utils, ops
from api.core.autograd.node import *
from api.core.autograd.graph import Graph, get_current_graph, reset_current_graph
from api.core.autograd.session import Session
