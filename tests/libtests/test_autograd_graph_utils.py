from api.lib import autograd as ag
from api.lib.autograd.node import (
    Variable, Add, Node, topological_sort, node_wrapper
)


def test_topological_sort():
    a = Variable(1, name='node1')
    b = Variable(2, name='node2')
    w = Variable(3, name='node3')

    c = ag.add(a, b, name='sum')
    x = ag.mul(c, b, name='mul')
    y = ag.add(x, w, name='sum1')
    # pay attention to the names of nodes and operators
    # to change something above make sure that expected
    # order will be changed too
    expected_order = ['node1', 'node2', 'sum', 'mul', 'node3', 'sum1']
    received = next(topological_sort(y))
    assert [n.name for n in received] == expected_order

    s = ag.add(a, b, name='sum2')
    s1 = ag.add(s, w, name='sum3')

    sorted_ = topological_sort([s, s1])

    received = next(sorted_)
    assert [n.name for n in received] == ['node1', 'node2', 'sum2']

    received = next(sorted_)
    assert [n.name for n in received] == ['node1', 'node2', 'sum2', 'node3', 'sum3']


def test_node_wrapper():
    x, c = Variable(1), 2

    node = Add(x, c)
    node_wrapped = node_wrapper(Add, x, c)
    node_wrapped_auto = x + c
    print(node_wrapped_auto)
    assert not all([isinstance(n, Node) for n in node.inputs])
    assert all([isinstance(n, Node) for n in node_wrapped.inputs])
    assert all([isinstance(n, Node) for n in node_wrapped_auto.inputs])
