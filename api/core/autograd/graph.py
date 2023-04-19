"""Contains definition of graph.

Computational graph is a form of directed acyclic graph that represents a
mathematical expression. In other words - tracker of simple operations. Thanks
to such graph, it becomes possible to apply Automatic Differentiation(AD).
"""
import itertools

# disabled W0603(global-statement) until stack will be implemented
# pylint: disable=W0603
# TODO: graph thread-safe stack to keep multiple graphs
_GRAPH = None  # var to store current graph


class Graph:
    """Computational graph class."""

    count = itertools.count().__next__

    def __init__(self):
        """Constructor method"""
        self.name = f"graph-{Graph.count()}"

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
