from api.core.autograd import ops
from api.core.autograd.node import Constant, Add, Node
from api.core.autograd.utils import topological_sort, node_wrapper


def test_topological_sort():
    a = Constant(1, name='node1')
    b = Constant(2, name='node2')
    w = Constant(3, name='node3')

    assert next(topological_sort(a))[0] is a

    c = ops.add(a, b, name='sum')
    x = ops.mul(c, b, name='mul')
    y = ops.add(x, w, name='sum1')

    def check_order(rcv, exp):
        return all([n == exp[i] for i, n in enumerate(rcv)])

    # pay attention to the names of nodes and operators
    # to change something above make sure that expected
    # order will be changed too
    expected_order = [a, b, c, x, w, y]
    received = next(topological_sort(y))
    assert check_order(received, expected_order)

    s = ops.add(a, b, name='sum2')
    s1 = ops.add(s, w, name='sum3')

    sorted_ = topological_sort([s, s1])

    expected_order = [a, b, s]
    received = next(sorted_)
    assert check_order(received, expected_order)

    expected_order = [a, b, s, w, s1]
    received = next(sorted_)
    assert check_order(received, expected_order)


def test_node_wrapper():
    x, c = Constant(1), 2

    node = Add(x, c)
    node_wrapped = node_wrapper(Add, x, c)
    node_wrapped_auto = x + c
    print(node_wrapped_auto)
    assert not all([isinstance(n, Node) for n in node.inputs])
    assert all([isinstance(n, Node) for n in node_wrapped.inputs])
    assert all([isinstance(n, Node) for n in node_wrapped_auto.inputs])
