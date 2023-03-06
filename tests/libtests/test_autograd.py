import pytest
import numpy as np
from api.lib import namespace, autograd as ag


sigmoid_autograd = lambda x: 1 / (1 + ag.ops.exp(-x))
sigmoid_numpy = lambda x: 1 / (1 + np.exp(np.negative(x)))
dsigmoid = lambda x: sigmoid_numpy(x) * (1 - sigmoid_numpy(x))

relu_autograd = lambda x: ag.ops.max(0, x)
relu_numpy = lambda x: np.maximum(0, x)
drelu = lambda x: np.where(x <= 0, 0, 1)

test_cases_fns = [
    (sigmoid_autograd, sigmoid_numpy, dsigmoid),
    (relu_autograd, relu_numpy, drelu)
]


@pytest.mark.parametrize('test_fn', test_cases_fns, ids=['sigmoid', 'relu'])
def test_autograd_using_complex_functions(session, test_fn, test_case_unary):
    autograd_fn, numpy_fn, d_numpy_fn = test_fn
    x = namespace.nodes.placeholder('x')
    out = autograd_fn(x)

    forward = session.run(out, feed_dict={x.name: iter([test_case_unary])})
    x_grd = session.gradients(out, [x])

    test_case = np.asarray(test_case_unary)
    assert np.allclose(numpy_fn(test_case), forward)
    assert np.allclose(d_numpy_fn(test_case), x_grd)
