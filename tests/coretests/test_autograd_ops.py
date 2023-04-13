import numpy as np

from api.core import namespace
from api.core.autograd import ops


def test_elementary_operations_binary(session, test_case_binary):
    tcb = test_case_binary
    assert np.array_equal(session.run(ops.add(*tcb)), np.add(*tcb))
    assert np.array_equal(session.run(ops.mul(*tcb)), np.multiply(*tcb))
    assert np.array_equal(session.run(ops.div(*tcb)), np.divide(*tcb))
    assert np.array_equal(session.run(ops.pow(*tcb)), np.power(*tcb))
    assert np.array_equal(session.run(ops.max(*tcb)), np.maximum(*tcb))
    assert np.array_equal(session.run(ops.min(*tcb)), np.minimum(*tcb))


def test_elementary_operations_unary(session, test_case_unary):
    tcu = test_case_unary
    assert np.array_equal(session.run(ops.sum(tcu)), np.sum(tcu))
    assert np.array_equal(
        session.run(ops.sum(tcu, axis=-1)), np.sum(tcu, axis=-1)
    )
    assert np.array_equal(session.run(ops.mean(tcu)), np.mean(tcu))
    if np.asarray(tcu).ndim >= 1:
        assert np.array_equal(
            session.run(ops.mean(tcu, axis=-1)), np.mean(tcu, axis=-1)
        )
    assert np.array_equal(session.run(ops.sqrt(tcu)), np.sqrt(tcu))
    assert np.array_equal(session.run(ops.rsqrt(tcu)), 1 / np.sqrt(tcu))
    assert np.array_equal(session.run(ops.abs(tcu)), np.abs(tcu))
    assert np.array_equal(session.run(ops.exp(tcu)), np.exp(tcu))
    assert np.array_equal(session.run(ops.log(tcu)), np.log(tcu))
    assert np.array_equal(session.run(ops.log2(tcu)), np.log2(tcu))
    assert np.array_equal(session.run(ops.log10(tcu)), np.log10(tcu))
    assert np.array_equal(session.run(ops.sin(tcu)), np.sin(tcu))
    assert np.array_equal(session.run(ops.cos(tcu)), np.cos(tcu))


def test_assign_op(session, test_case_binary):
    tcb = test_case_binary

    x = namespace.nodes.variable(tcb[0])
    expected = np.add(*tcb)
    assert np.array_equal(session.run(ops.assign_add(x, tcb[1])), expected)
    assert np.array_equal(x.value, expected)

    x = namespace.nodes.variable(tcb[0])
    assert np.array_equal(session.run(ops.assign(x, tcb[1])), tcb[1])
    assert np.array_equal(x.value, tcb[1])

    x = namespace.nodes.variable(tcb[0])
    expected = np.multiply(*tcb)
    assert np.array_equal(session.run(ops.assign_mul(x, tcb[1])), expected)
    assert np.array_equal(x.value, expected)

    x = namespace.nodes.variable(tcb[0])
    expected = np.divide(*tcb)
    assert np.array_equal(session.run(ops.assign_div(x, tcb[1])), expected)
    assert np.array_equal(x.value, expected)


def test_other_ops(session):
    a, b = np.ones((3, 1, 1, 2)), np.ones((2, 3))
    a_var = namespace.nodes.variable(a)

    subscr = "ijhw, wk -> ijhk"
    expected = np.einsum(subscr, a, b)
    assert np.array_equal(session.run(ops.einsum(subscr, a_var, b)), expected)
    assert session.run(ops.einsum(subscr, a_var, b)).shape == (3, 1, 1, 3)
