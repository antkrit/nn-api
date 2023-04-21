import re

import numpy as np
import pytest

from api.core.autograd.graph import Graph
from api.core.autograd.node import *
from tests.utils import check_node_name_format, element_wise_equal


def test_base_node():
    n = Node(value=None, name=None, shape=None)
    assert hasattr(n, "count")
    assert hasattr(n, "current_graph")

    pgraph = n.current_graph()
    assert isinstance(pgraph, Graph)

    g = Graph()
    g.as_default()
    cgraph = n.current_graph()
    assert cgraph is g
    assert cgraph is not pgraph

    assert n.value is None

    _ = Node(value=[1], name=None, shape=(1,))
    n = Node(value=[1], name=None, shape=None)
    assert n.shape == (1,)
    assert check_node_name_format(n)


def test_variable_node(session):
    v_val, v_name = 3, "v1"
    var = Variable(v_val)
    assert var.value == v_val and isinstance(var.name, str)
    assert var.shape == var.value.shape

    vrbl_rgx = re.compile(r"^.*/variable-\d+$")
    assert re.search(vrbl_rgx, var.name) is not None

    var.name = v_name
    assert str(var) == v_name

    _ = Variable(1, shape=())
    _ = Variable([1, 2], shape=(2,))


def test_constant_node():
    c_val, c_name = 3, "c1"
    cnst = Constant(c_val)
    assert cnst.value == c_val and isinstance(cnst.name, str)
    assert cnst.shape == cnst.value.shape

    cnst_rgx = re.compile(r"^.*/constant-\d+$")
    assert re.search(cnst_rgx, cnst.name) is not None

    cnst.name = c_name
    assert str(cnst) == c_name

    with pytest.raises(ValueError):
        cnst.value = 1

    _ = Constant(1, shape=())
    _ = Constant([1, 2], shape=(2,))


def test_placeholder_node():
    p_val, p_name = 3, "p1"
    plc = Placeholder(p_name)
    assert plc.value is None and isinstance(plc.name, str)

    plc.name = p_name
    assert str(plc) == p_name

    plc = Placeholder()
    plc_rgx = re.compile(r"^.*/placeholder-\d+$")
    assert re.search(plc_rgx, plc.name) is not None

    shape = (3, 4)
    plc_with_shape = Placeholder(name=p_name, shape=shape)
    plc_with_shape.value = np.ones(shape)
    assert np.array_equal(plc_with_shape.value, np.ones(shape))


def test_operation_nodes():
    op_name = "op1"
    op = Operation()

    oprtr_rgx = re.compile(r"^.*/operator-\d+$")
    assert re.search(oprtr_rgx, op.name) is not None

    op.name = op_name
    assert str(op) == op_name

    with pytest.raises(NotImplementedError):
        op.forward()

    with pytest.raises(NotImplementedError):
        op.backward()

    shape = (4, 3)
    test_value = np.ones(shape)
    op.value = test_value
    assert np.array_equal(op.value, test_value)
    assert op.shape == test_value.shape


dout_cases = [2, [1, 2, 3], [[1, 2, 3], [1, 2, 3]]]
dout_ids = ["dout_scalar", "dout_vector", "dout_matrix"]


class TestOtherOperations:
    def test_einsum(self):
        a = np.ones((3, 1, 2))
        b = np.ones((2, 1))
        expected_output_shape = (3, 1, 1)
        expected_subscript = "bhw, wk->bhk"

        es = Einsum(a, b, subscripts=["bhw", "wk"], o_subscript="bhk")

        assert es.subscripts() == expected_subscript
        assert np.array_equal(
            es.forward(*es.inputs), np.einsum(expected_subscript, a, b)
        )
        assert es.forward(*es.inputs).shape == expected_output_shape
        a_grd, b_grd = es.backward(
            *es.inputs, dout=np.zeros(expected_output_shape)
        )
        assert a_grd.shape == a.shape
        assert b_grd.shape == b.shape

        a = np.array([[1, 2], [3, 4]])
        es = Einsum(a, subscripts=["ij"], o_subscript="ji")
        assert np.array_equal(es.forward(*es.inputs), np.transpose(a))
        assert es.backward(a, dout=np.zeros(a.shape)).shape == a.shape

    @pytest.mark.parametrize("dout", dout_cases, ids=dout_ids)
    @pytest.mark.parametrize(
        "test_case",
        [
            (np.random.randint(1, 10), np.random.randint(1, 10)),
            (np.random.randint(1, 10, size=(3,)), np.random.randint(1, 10)),
            (np.random.randint(1, 10, size=(2, 3)), np.random.randint(1, 10)),
        ],
        ids=["scalar^scalar", "vector^scalar", "matrix^scalar"],
    )
    def test_power(self, test_case, dout):
        a, b = test_case
        p = Power(a, b)
        assert np.array_equal(p.forward(a, b), np.power(a, b))
        assert np.allclose(
            p.backward(a, b, dout),
            [
                np.multiply(
                    dout, np.asarray(b) * np.power(a, np.asarray(b) - 1)
                ),
                np.multiply(dout, np.power(a, b) * np.log(a)),
            ],
        )

    @pytest.mark.parametrize(
        "test_case, axis, dout",
        [
            [1, None, 9],
            [[1, 2, 3], None, 2],
            [[1, 2, 3], 0, 4],
            [[[2, 2, 2], [1, 2, 3]], None, -5],
            [[[2, 2, 2], [1, 2, 3]], 0, [[2, 2, 2]]],
            [[[2, 2, 2], [1, 2, 3]], 1, [[3], [3]]],
        ],
    )
    def test_sum(self, test_case, axis, dout):
        a, acopy = test_case, np.asarray(test_case)
        s = Sum(a, axis=axis)
        assert np.array_equal(s.forward(a), np.sum(a, axis))
        assert np.array_equal(
            s.backward(a, dout=dout), np.multiply(dout, np.ones(acopy.shape))
        )

    @pytest.mark.parametrize(
        "test_case, axis, dout",
        [
            [1, None, 9],
            [[1, 2, 3], None, 2],
            [[1, 2, 3], 0, 4],
            [[[2, 2, 2], [1, 2, 3]], None, 5],
            [[[2, 2, 2], [1, 2, 3]], 0, [[2, 2, 2]]],
            [[[2, 2, 2], [1, 2, 3]], 1, [[3], [3]]],
        ],
    )
    def test_mean(self, test_case, axis, dout):
        a, acopy = test_case, np.asarray(test_case)
        m = Mean(a, axis=axis)
        assert np.array_equal(m.forward(a), np.mean(a, axis))
        assert np.allclose(
            m.backward(a, dout=dout),
            np.multiply(dout, np.divide(np.ones(acopy.shape), acopy.size)),
        )


@pytest.mark.parametrize("dout", dout_cases, ids=dout_ids)
class TestUnaryOperators:
    def test_reshape(self, test_case_unary, dout):
        a = np.asarray(test_case_unary)
        to_shape = a.size
        s = Reshape(a, to_shape=a.size)
        assert np.array_equal(s.forward(a), np.reshape(a, to_shape))
        assert np.array_equal(s.backward(a, dout), [np.multiply(dout, a)])

    def test_flatten(self, test_case_unary, dout):
        a = np.asarray(test_case_unary)
        s = Flatten(a)
        assert np.array_equal(s.forward(a), a.flatten())
        assert np.array_equal(s.backward(a, dout), [np.multiply(dout, a)])

    def test_sqrt(self, test_case_unary, dout):
        a = test_case_unary
        s = Sqrt(a)
        assert np.array_equal(s.forward(a), np.sqrt(a))
        assert np.array_equal(
            s.backward(a, dout), [dout * 1 / (2 * np.sqrt(a))]
        )

    def test_abs(self, test_case_unary, dout):
        a = test_case_unary
        abs_ = Abs(a)
        assert np.array_equal(abs_.forward(a), np.abs(a))
        assert np.array_equal(
            abs_.backward(a, dout), [np.multiply(dout, (a / (np.abs(a))))]
        )

    def test_exp(self, test_case_unary, dout):
        a = test_case_unary
        e = Exp(a)
        assert np.allclose(e.forward(a), np.exp(a))
        assert np.allclose(
            e.backward(a, dout), [np.multiply(dout, np.exp(a))], rtol=1e-9
        )

    def test_log(self, test_case_unary, dout):
        a = test_case_unary
        e = Log(a)
        assert np.array_equal(e.forward(a), np.log(a))
        assert np.array_equal(e.backward(a, dout), [dout * 1 / np.asarray(a)])

    def test_log2(self, test_case_unary, dout):
        a = test_case_unary
        e = Log2(a)
        assert np.array_equal(e.forward(a), np.log2(a))
        assert np.array_equal(
            e.backward(a, dout), [dout * 1 / (np.asarray(a) * np.log(2))]
        )

    def test_log10(self, test_case_unary, dout):
        a = test_case_unary
        e = Log10(a)
        assert np.array_equal(e.forward(a), np.log10(a))
        assert np.array_equal(
            e.backward(a, dout), [dout * 1 / (np.asarray(a) * np.log(10))]
        )

    def test_sin(self, test_case_unary, dout):
        a = test_case_unary
        s = Sin(a)
        assert np.array_equal(s.forward(a), np.sin(a))
        assert np.array_equal(
            s.backward(a, dout), [np.multiply(dout, np.cos(a))]
        )

    def test_cos(self, test_case_unary, dout):
        a = test_case_unary
        c = Cos(a)
        assert np.array_equal(c.forward(a), np.cos(a))
        assert np.array_equal(
            c.backward(a, dout), [np.multiply(dout, (-np.sin(a)))]
        )


@pytest.mark.parametrize("dout", [1.0], ids=["dout_scalar"])
@pytest.mark.parametrize(
    "test_case",
    [
        (np.random.randint(1, 10), np.random.randint(1, 10)),
        (
            np.random.randint(1, 10, size=(3,)),
            np.random.randint(1, 10, size=(3,)),
        ),
        (
            np.random.randint(1, 10, size=(2, 2)),
            np.random.randint(1, 10, size=(2, 2)),
        ),
        (
            np.random.randint(1, 10, size=(4, 1, 2)),
            np.random.randint(1, 10, size=(2, 2)),
        ),
    ],
)
class TestBinaryOperators:
    def test_add(self, test_case, dout):
        a, b = test_case
        add = Add(a, b)
        assert np.array_equal(add.forward(a, b), np.asarray(a) + np.asarray(b))
        assert np.array_equal(
            add.backward(a, b, dout),
            [np.multiply(dout, 1), np.multiply(dout, 1)],
        )

    def test_assign_add(self, test_case, dout):
        a, b = test_case
        a_var = Variable(a)

        with pytest.raises(ValueError):
            AssignAdd(a, b)

        a_add = AssignAdd(a_var, b)
        assert np.array_equal(
            a_add.forward(a, b), np.asarray(a) + np.asarray(b)
        )
        assert np.array_equal(a_add.backward(a, b, dout), [dout * 1, dout * 1])

    def test_assign(self, test_case, dout):
        a, b = test_case
        a_var = Variable(a)

        with pytest.raises(ValueError):
            Assign(a, b)

        a_add = Assign(a_var, b)
        assert np.array_equal(
            a_add.forward(b, 0), np.asarray(b) + np.asarray(0)
        )
        assert np.array_equal(a_add.backward(a, b, dout), [dout * 1, dout * 1])

    def test_multiply(self, test_case, dout):
        a, b = test_case
        mul = Multiply(a, b)
        assert np.array_equal(mul.forward(a, b), np.array(a) * np.array(b))
        assert element_wise_equal(
            mul.backward(a, b, dout),
            [np.multiply(dout, b), np.multiply(dout, a)],
        )

    def test_assign_multiply(self, test_case, dout):
        a, b = test_case
        a_var = Variable(a)

        with pytest.raises(ValueError):
            AssignMultiply(a, b)

        a_mul = AssignMultiply(a_var, b)
        assert np.array_equal(a_mul.forward(a, b), np.array(a) * np.array(b))
        assert element_wise_equal(
            a_mul.backward(a, b, dout),
            [np.multiply(dout, b), np.multiply(dout, a)],
        )

    def test_divide(self, test_case, dout):
        a, b = test_case
        div = Divide(a, b)

        assert np.array_equal(div.forward(a, b), np.array(a) / np.array(b))
        assert element_wise_equal(
            div.backward(a, b, dout),
            [
                np.divide(np.asarray(dout), b),
                np.multiply(np.negative(dout), np.divide(a, np.power(b, 2))),
            ],
        )

    def test_assign_divide(self, test_case, dout):
        a, b = test_case
        a_var = Variable(a)

        with pytest.raises(ValueError):
            AssignDivide(a, b)

        a_div = AssignDivide(a_var, b)
        assert np.array_equal(a_div.forward(a, b), np.array(a) / np.array(b))
        assert element_wise_equal(
            a_div.backward(a, b, dout),
            [
                np.divide(np.asarray(dout), np.array(b)),
                np.multiply(np.negative(dout), np.divide(a, np.power(b, 2))),
            ],
        )

    def test_matmul(self, test_case, dout):
        a, b = test_case
        mm = Matmul(a, b)
        assert np.array_equal(mm.forward(a, b), np.asarray(a).dot(b))
        assert element_wise_equal(
            mm.backward(a, b, dout),
            [np.dot(dout, np.asarray(b).T), np.dot(np.asarray(a).T, dout)],
        )

    def test_max(self, test_case, dout):
        a, b = test_case
        m = Max(a, b)
        assert np.array_equal(m.forward(a, b), np.maximum(a, b))
        assert np.array_equal(
            m.backward(a, b, dout),
            [dout * np.where(a > b, 1, 0), dout * np.where(a > b, 0, 1)],
        )

    def test_min(self, test_case, dout):
        a, b = test_case
        m = Min(a, b)
        assert np.array_equal(m.forward(a, b), np.minimum(a, b))
        assert np.array_equal(
            m.backward(a, b, dout),
            [dout * np.where(a < b, 1, 0), dout * np.where(a < b, 0, 1)],
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
    assert session.run(a**b) == 1
    assert session.run(b**a) == 2

    a = [[1, 1], [2, 2]]
    b = [[2, 2], [1, 1]]
    assert np.array_equal(session.run(Variable(a) @ b), np.dot(a, b))
