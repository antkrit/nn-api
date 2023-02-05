import re
import pytest
import numpy as np
from api.lib.autograd.graph import Graph
from api.lib.autograd.node import (
    Node, Constant, Variable, Placeholder, Operation,
    Add, Multiply, Divide, Power, Matmul, Max, Min, Sum, Mean,
    Sqrt, Abs, Exp, Log, Log2, Log10, Cos, Sin
)


def test_base_node():
    n = Node()
    assert hasattr(n, 'count')
    assert hasattr(n, 'graph')

    ograph = n.graph()
    assert isinstance(ograph, Graph)

    g = Graph()
    g.as_default()
    cgraph = n.graph()
    assert cgraph is g
    assert cgraph is not ograph


def test_variable_node():
    v_val, v_name = 3, 'v1'
    v = Variable(v_val)
    assert v.value == v_val and isinstance(v.name, str)

    vrbl_rgx = re.compile(r'^.*/variable-\d+$')
    assert re.search(vrbl_rgx, v.name) is not None

    v.name = v_name
    assert str(v) == v_name


def test_constant_node():
    c_val, c_name = 3, 'c1'
    c = Constant(c_val)
    assert c.value == c_val and isinstance(c.name, str)

    cnst_rgx = re.compile(r'^.*/constant-\d+$')
    assert re.search(cnst_rgx, c.name) is not None

    c.name = c_name
    assert str(c) == c_name
    with pytest.raises(ValueError):
        c.value = 1


def test_placeholder_node():
    p_val, p_name = 3, 'p1'
    p = Placeholder(p_name)
    assert p.value is None and p.name == p_name

    with pytest.raises(ValueError):
        Placeholder(name=None)

    p.name = p_name
    assert str(p) == p_name


def test_operation_nodes():
    op_name = 'op1'
    op = Operation(op_name)

    oprtr_rgx = re.compile(rf'^.*/operator-{op_name}-\d+$')
    assert re.search(oprtr_rgx, op.name) is not None

    with pytest.raises(NotImplementedError):
        op.forward()

    with pytest.raises(NotImplementedError):
        op.backward()


dout_cases = [2, [1, 2, 3], [[1, 2, 3], [1, 2, 3]]]
dout_ids = ['dout_scalar', 'dout_vector', 'dout_matrix']


class TestOtherOperations:

    @pytest.mark.parametrize('dout', dout_cases, ids=dout_ids)
    @pytest.mark.parametrize('test_case', [
        (3, 7),
        ([1, 2, 3], 3),
        ([[2, 3, 4], [2, 3, 4]], 2)
    ], ids=['scalar^scalar', 'vector^scalar', 'matrix^scalar'])
    def test_power(self, test_case, dout):
        a, b = test_case
        p = Power(a, b)
        assert np.array_equal(p.forward(a, b), np.power(a, b))
        assert np.allclose(
            p.backward(a, b, dout),
            [
                np.multiply(dout, np.asarray(b) * np.power(a, np.asarray(b)-1)),
                np.multiply(dout, np.power(a, b) * np.log(a))
            ]
        )

    @pytest.mark.parametrize('test_case, axis, dout', [
        [1, None, 9],
        [[1, 2, 3], None, 2],
        [[1, 2, 3], 0, 4],
        [[[2, 2, 2], [1, 2, 3]], None, -5],
        [[[2, 2, 2], [1, 2, 3]], 0, [[2, 2, 2]]],
        [[[2, 2, 2], [1, 2, 3]], 1, [[3], [3]]],
    ])
    def test_sum(self, test_case, axis, dout):
        a, acopy = test_case, np.asarray(test_case)
        s = Sum(a, axis=axis)
        assert np.array_equal(s.forward(a), np.sum(a, axis))
        assert np.array_equal(
            s.backward(a, dout=dout),
            np.multiply(dout, np.ones(acopy.shape))
        )

    @pytest.mark.parametrize('test_case, axis, dout', [
        [1, None, 9],
        [[1, 2, 3], None, 2],
        [[1, 2, 3], 0, 4],
        [[[2, 2, 2], [1, 2, 3]], None, 5],
        [[[2, 2, 2], [1, 2, 3]], 0, [[2, 2, 2]]],
        [[[2, 2, 2], [1, 2, 3]], 1, [[3], [3]]],
    ])
    def test_sum(self, test_case, axis, dout):
        a, acopy = test_case, np.asarray(test_case)
        m = Mean(a, axis=axis)
        assert np.array_equal(m.forward(a), np.mean(a, axis))
        assert np.allclose(
            m.backward(a, dout=dout),
            np.multiply(dout, np.divide(np.ones(acopy.shape), acopy.size))
        )


@pytest.mark.parametrize('dout', dout_cases, ids=dout_ids)
class TestUnaryOperators:

    def test_sqrt(self, test_case_unary, dout):
        a = test_case_unary
        s = Sqrt(a)
        assert np.array_equal(s.forward(a), np.sqrt(a))
        assert np.array_equal(
            s.backward(a, dout),
            [dout * 1 / (2*np.sqrt(a))]
        )

    def test_abs(self, test_case_unary, dout):
        a = test_case_unary
        abs_ = Abs(a)
        assert np.array_equal(abs_.forward(a), np.abs(a))
        assert np.array_equal(
            abs_.backward(a, dout),
            [np.multiply(dout, (a / (np.abs(a))))]
        )

    def test_exp(self, test_case_unary, dout):
        a = test_case_unary
        e = Exp(a)
        assert np.allclose(e.forward(a), np.exp(a))
        assert np.allclose(
            e.backward(a, dout),
            [np.multiply(dout, np.exp(a))],
            rtol=1e-9
        )

    def test_log(self, test_case_unary, dout):
        a = test_case_unary
        e = Log(a)
        assert np.array_equal(e.forward(a), np.log(a))
        assert np.array_equal(
            e.backward(a, dout),
            [dout * 1/np.asarray(a)]
        )

    def test_log2(self, test_case_unary, dout):
        a = test_case_unary
        e = Log2(a)
        assert np.array_equal(e.forward(a), np.log2(a))
        assert np.array_equal(
            e.backward(a, dout),
            [dout * 1 / (np.asarray(a)*np.log(2))]
        )

    def test_log10(self, test_case_unary, dout):
        a = test_case_unary
        e = Log10(a)
        assert np.array_equal(e.forward(a), np.log10(a))
        assert np.array_equal(
            e.backward(a, dout),
            [dout * 1 / (np.asarray(a) * np.log(10))]
        )

    def test_sin(self, test_case_unary, dout):
        a = test_case_unary
        s = Sin(a)
        assert np.array_equal(s.forward(a), np.sin(a))
        assert np.array_equal(
            s.backward(a, dout),
            [np.multiply(dout, np.cos(a))]
        )

    def test_cos(self, test_case_unary, dout):
        a = test_case_unary
        c = Cos(a)
        assert np.array_equal(c.forward(a), np.cos(a))
        assert np.array_equal(
            c.backward(a, dout),
            [np.multiply(dout, (-np.sin(a)))]
        )


@pytest.mark.parametrize('dout', [2], ids=['scalar'])
@pytest.mark.parametrize('test_case', [
    (-4, 5),
    (np.array([1, -2, 3]), [2, 3, 4]),
    ([[1, -2], [-1, 2]], np.array([[2, 3], [2, 3]])),
])
class TestBinaryOperators:

    def test_add(self, test_case, dout):
        a, b = test_case
        add = Add(a, b)
        assert np.array_equal(
            add.forward(a, b),
            np.asarray(a) + np.asarray(b)
        )
        assert np.array_equal(
            add.backward(a, b, dout),
            [dout*1, dout*1]
        )

    def test_multiply(self, test_case, dout):
        a, b = test_case
        m = Multiply(a, b)
        assert np.array_equal(
            m.forward(a, b),
            np.array(a) * np.array(b)
        )
        assert np.array_equal(
            m.backward(a, b, dout),
            [dout * np.array(b), dout * np.array(a)]
        )

    def test_divide(self, test_case, dout):
        a, b = test_case
        d = Divide(a, b)
        assert np.array_equal(
            d.forward(a, b),
            np.array(a) / np.array(b)
        )
        assert np.array_equal(
            d.backward(a, b, dout),
            [
                np.asarray(dout) / np.array(b),
                np.negative(dout) * np.array(a) / np.power(np.array(b), 2)
            ]
        )

    def test_matmul(self, test_case, dout):
        a, b = test_case
        mm = Matmul(a, b)
        assert np.array_equal(mm.forward(a, b), np.asarray(a).dot(b))
        assert np.array_equal(
            mm.backward(a, b, dout),
            [
                np.dot(dout, np.asarray(b).T),
                np.dot(np.asarray(a).T, dout)
            ]
        )

    def test_max(self, test_case, dout):
        a, b = test_case
        m = Max(a, b)
        assert np.array_equal(m.forward(a, b), np.maximum(a, b))
        assert np.array_equal(
            m.backward(a, b, dout),
            [
                dout * np.where(a > b, 1, 0),
                dout * np.where(a > b, 0, 1)
            ]
        )

    def test_min(self, test_case, dout):
        a, b = test_case
        m = Min(a, b)
        assert np.array_equal(m.forward(a, b), np.minimum(a, b))
        assert np.array_equal(
            m.backward(a, b, dout),
            [
                dout * np.where(a < b, 1, 0),
                dout * np.where(a < b, 0, 1)
            ]
        )


def test_dunder_methods(session):

    a = Constant(1)
    b = 2
    assert session.run(a + b) == session.run(b + a) == 3
    assert session.run(a - b) == -1
    assert session.run(b - a) == 1
    assert session.run(a * b) == session.run(b * a) == 2
    assert session.run(a / b) == 0.5
    assert session.run(b / a) == 2
    assert session.run(a ** b) == 1
    assert session.run(b ** a) == 2

    a = [[1, 1], [2, 2]]
    b = [[2, 2], [1, 1]]
    assert np.array_equal(session.run(Variable(a) @ b), np.dot(a, b))
