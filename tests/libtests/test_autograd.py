import pytest
import numpy as np
import api.lib.autograd as ag
from api.lib.autograd import Placeholder


sigmoid_autograd = lambda x: 1 / (1 + ag.exp(-x))
sigmoid_numpy = lambda x: 1 / (1 + np.exp(np.negative(x)))
dsigmoid = lambda x: sigmoid_numpy(x) * (1 - sigmoid_numpy(x))

relu_autograd = lambda x: ag.max(0, x)
relu_numpy = lambda x: np.maximum(0, x)
drelu = lambda x: np.where(x <= 0, 0, 1)

test_cases_fns = [
    (sigmoid_autograd, sigmoid_numpy, dsigmoid),
    (relu_autograd, relu_numpy, drelu)
]


@pytest.mark.parametrize('test_fn', test_cases_fns, ids=['sigmoid', 'relu'])
def test_autograd_using_complex_functions(session, test_fn, test_case_unary):
    autograd_fn, numpy_fn, d_numpy_fn = test_fn
    x = Placeholder('x')
    out = autograd_fn(x)

    frwrd = session.run(out, feed_dict={'x': test_case_unary})
    grads = session.gradients(out)

    test_case = np.asarray(test_case_unary)
    assert np.allclose(numpy_fn(test_case), frwrd)
    assert np.allclose(d_numpy_fn(test_case), grads[x])


def test_elementary_operations_binary(session, test_case_binary):
    tc = test_case_binary
    assert np.array_equal(session.run(ag.add(*tc)), np.add(*tc))
    assert np.array_equal(session.run(ag.mul(*tc)), np.multiply(*tc))
    assert np.array_equal(session.run(ag.div(*tc)), np.divide(*tc))
    assert np.array_equal(session.run(ag.pow(*tc)), np.power(*tc))
    assert np.array_equal(session.run(ag.max(*tc)), np.maximum(*tc))
    assert np.array_equal(session.run(ag.min(*tc)), np.minimum(*tc))


def test_elementary_operations_unary(session, test_case_unary):
    tc = test_case_unary
    assert np.array_equal(session.run(ag.sum(tc)), np.sum(tc))
    assert np.array_equal(
        session.run(ag.sum(tc, axis=-1)), np.sum(tc, axis=-1)
    )
    assert np.array_equal(session.run(ag.mean(tc)), np.mean(tc))
    if np.asarray(tc).ndim >= 1:
        assert np.array_equal(
            session.run(ag.mean(tc, axis=-1)), np.mean(tc, axis=-1)
        )
    assert np.array_equal(session.run(ag.sqrt(tc)), np.sqrt(tc))
    assert np.array_equal(session.run(ag.abs(tc)), np.abs(tc))
    assert np.array_equal(session.run(ag.exp(tc)), np.exp(tc))
    assert np.array_equal(session.run(ag.log(tc)), np.log(tc))
    assert np.array_equal(session.run(ag.log2(tc)), np.log2(tc))
    assert np.array_equal(session.run(ag.log10(tc)), np.log10(tc))
    assert np.array_equal(session.run(ag.sin(tc)), np.sin(tc))
    assert np.array_equal(session.run(ag.cos(tc)), np.cos(tc))
