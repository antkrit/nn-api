"""Contain a `math` object with all traceable mathematical operations."""
import inspect
from api.lib.autograd.graph import Node
from api.lib.autograd._util import dotdict


_math_ops = inspect.getmembers(Node, predicate=inspect.isfunction)
_math_ops.extend(inspect.getmembers(Node, predicate=inspect.isclass))
math = dotdict(_math_ops)
