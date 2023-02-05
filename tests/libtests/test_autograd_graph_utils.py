from api.lib.autograd.node import (
    Variable, Add, Node, topological_sort, node_wrapper
)


def test_topological_sort():
    a = Variable(1, name='node1')
    b = Variable(2, name='node2')
    w = Variable(3, name='node3')

    c, c.name = a + b, 'sum'
    x, x.name = c * b, 'mul'
    y, y.name = x + w, 'sum1'
    # pay attention to the names of nodes and operators
    # to change something above make sure that expected
    # order will be changed too
    expected_order = ['node1', 'node2', 'sum', 'mul', 'node3', 'sum1']
    assert list((n.name for n in topological_sort(y))) == expected_order

    s, s.name = a + b, 'sum2'
    s1, s1.name = s + w, 'sum3'
    s2, s2.name = s1 + s, 'sum4'
    expected_order = ['node1', 'node2', 'sum2', 'node3', 'sum3', 'sum4']
    assert list((n.name for n in topological_sort(s2))) == expected_order


def test_node_wrapper():
    x, c = Variable(1), 2

    node = Add(x, c)
    node_wrapped = node_wrapper(Add, x, c)
    node_wrapped_auto = x + c

    assert not all([isinstance(n, Node) for n in node.inputs])
    assert all([isinstance(n, Node) for n in node_wrapped.inputs])
    assert all([isinstance(n, Node) for n in node_wrapped_auto.inputs])
