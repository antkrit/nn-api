import numpy as np
import pytest

import api.core.activation as activation
import api.core.autograd as ag

TEST_CASES = [0, [0, 1, 50, 100], [[0, 100], [-100, 0]]]
TEST_IDS = ["scalar", "vector", "matrix"]


def test_docstring_example(session, test_case_unary):
    class Linear(activation.BaseActivation):
        def forward(self, x):
            return x

    actv = Linear()
    out = session.run(actv(test_case_unary))
    assert np.array_equal(out, test_case_unary)


class TestSigmoid:
    backward_expected = [0.25, [0.25, 0.19661193, 0, 0], [[0.25, 0], [0, 0.25]]]

    @pytest.mark.parametrize("x", TEST_CASES, ids=TEST_IDS)
    def test_forward(self, session, x):
        a = activation.Sigmoid(session=session)

        assert np.allclose(session.run(a(x)), np.exp(x) / (1 + np.exp(x)))
        assert np.array_equal(
            session.run(a(x)), session.run(a.forward(ag.Constant(x)))
        )

    @pytest.mark.parametrize(
        "x, expected", zip(TEST_CASES, backward_expected), ids=TEST_IDS
    )
    def test_gradient(self, session, graph, x, expected):
        s = activation.Sigmoid(session=session)

        x = ag.Constant(x)
        y = s(x)
        _ = session.run(y)

        assert np.allclose(session.gradients(y, [x]), expected)


class TestTanh:
    forward_expected = [0, [0, 0.7615942, 1, 1], [[0, 1], [-1, 0]]]
    backward_expected = [1, [1, 0.41997433, 0, 0], [[1, 0], [0, 1]]]

    @pytest.mark.parametrize(
        "x, expected", zip(TEST_CASES, forward_expected), ids=TEST_IDS
    )
    def test_forward(self, session, x, expected):
        a = activation.Tanh(session=session)

        assert np.allclose(session.run(a(x)), expected)
        assert np.array_equal(
            session.run(a(x)), session.run(a.forward(ag.Constant(x)))
        )

    @pytest.mark.parametrize(
        "x, expected", zip(TEST_CASES, backward_expected), ids=TEST_IDS
    )
    def test_gradient(self, session, graph, x, expected):
        a = activation.Tanh(session=session)

        x = ag.Constant(x)
        y = a(x)
        _ = session.run(y)

        assert np.allclose(session.gradients(y, [x]), expected)


class TestReluLike:
    @pytest.mark.parametrize(
        "alpha",
        [0, 0.01, 0.05],
        ids=["not_leaky", "leaky_a=0.01", "leaky_a=0.05"],
    )
    @pytest.mark.parametrize("x", TEST_CASES, ids=TEST_IDS)
    def test_relu_forward(self, session, x, alpha):
        a = activation.ReLU(alpha=alpha, session=session)

        assert np.allclose(
            session.run(a(x)),
            np.where(np.asarray(x) >= 0, x, np.multiply(x, alpha)),
        )
        assert np.array_equal(
            session.run(a(x)), session.run(a.forward(ag.Constant(x)))
        )

    @pytest.mark.parametrize(
        "alpha",
        [0, 0.01, 0.05],
        ids=["not_leaky", "leaky_a=0.01", "leaky_a=0.05"],
    )
    @pytest.mark.parametrize("x", TEST_CASES, ids=TEST_IDS)
    def test_relu_gradient(self, session, graph, x, alpha):
        a = activation.ReLU(alpha=alpha, session=session)

        x_node = ag.Constant(x)
        y = a(x_node)
        _ = session.run(y)

        x = np.asarray(x)
        der = np.ones(x.shape)
        assert np.allclose(
            session.gradients(y, [x_node]),
            np.where(x > 0, der, np.multiply(der, alpha)),
        )

    @pytest.mark.parametrize(
        "alpha", [0.01, 0.05], ids=["leaky_a=0.01", "leaky_a=0.05"]
    )
    @pytest.mark.parametrize("x", TEST_CASES, ids=TEST_IDS)
    def test_elu_forward(self, session, x, alpha):
        a = activation.ELU(alpha=alpha, session=session)

        assert np.allclose(
            session.run(a(x)),
            np.where(np.asarray(x) >= 0, x, np.multiply(alpha, np.exp(x) - 1)),
        )
        assert np.array_equal(
            session.run(a(x)), session.run(a.forward(ag.Constant(x)))
        )

    @pytest.mark.parametrize(
        "alpha", [0.01, 0.05], ids=["leaky_a=0.01", "leaky_a=0.05"]
    )
    @pytest.mark.parametrize("x", TEST_CASES, ids=TEST_IDS)
    def test_elu_gradient(self, session, x, alpha):
        a = activation.ELU(alpha=alpha, session=session)

        x_node = ag.Constant(x)
        y = a(x_node)
        _ = session.run(y)

        x = np.asarray(x)
        der = np.ones(x.shape)
        assert np.allclose(
            session.gradients(y, [x_node]),
            np.where(x > 0, der, np.multiply(alpha, np.exp(x))),
        )


@pytest.mark.parametrize(
    "x, expected",
    [
        ([1.3, 5.1, 2.2, 0.7, 1.1], [0.02, 0.9, 0.05, 0.01, 0.02]),
        ([[1.5, 1.5], [1.5, 1.5]], [[0.25, 0.25], [0.25, 0.25]]),
        pytest.param(
            [1000, 2000, 3000],
            [0, 0, 1],
            marks=pytest.mark.xfail(
                reason="add shift to x to stabilize softmax in case of large "
                "data. See `api.core.activation.Softmax.forward()`."
            ),
        ),
    ],
    ids=["x_not_equal", "x_equal", "large_x"],
)
class TestSoftmax:
    def test_forward(self, session, x, expected):
        a = activation.Softmax(session=session)

        assert np.allclose(session.run(a(x)), expected, atol=0.01)
        assert np.array_equal(
            session.run(a(x)), session.run(a.forward(ag.Constant(x)))
        )

    def test_gradient(self, session, x, expected):
        a = activation.Softmax(session=session)

        x_node = ag.Constant(x)
        y = a(x_node)
        _ = session.run(y)

        assert np.allclose(
            session.gradients(y, [x_node]), np.zeros(np.asarray(x).shape)
        )


class TestSwish:
    forward_expected = [0, [0, 0.73, 50, 100], [[0, 100], [0, 0]]]
    backward_expected = [0.5, [0.5, 0.92, 1, 1], [[0.5, 1], [0, 0.5]]]

    @pytest.mark.parametrize("beta", [0, 1], ids=["beta=0", "beta=1"])
    @pytest.mark.parametrize(
        "x, expected", zip(TEST_CASES, forward_expected), ids=TEST_IDS
    )
    def test_forward(self, session, x, expected, beta):
        a = activation.Swish(beta=beta, session=session)

        assert np.allclose(
            session.run(a(x)),
            # note: this expected value hardcoded to fit parametrize data
            expected if beta == 1 else np.asarray(x) / 2,
            atol=0.01,
        )
        assert np.array_equal(
            session.run(a(x)), session.run(a.forward(ag.Constant(x)))
        )

    @pytest.mark.parametrize(
        "x, expected", zip(TEST_CASES, backward_expected), ids=TEST_IDS
    )
    def test_gradient(self, session, x, expected):
        a = activation.Swish(session=session)

        x_node = ag.Constant(x)
        y = a(x_node)
        _ = session.run(y)

        assert np.allclose(session.gradients(y, [x_node]), expected, atol=0.01)


class TestSoftplus:
    forward_expected = [0.69, [0.69, 1.31, 50, 100], [[0.69, 100], [0, 0.69]]]
    backward_expected = [0.5, [0.5, 0.73, 1, 1], [[0.5, 1], [0, 0.5]]]

    @pytest.mark.parametrize(
        "x, expected", zip(TEST_CASES, forward_expected), ids=TEST_IDS
    )
    def test_forward(self, session, x, expected):
        a = activation.Softplus(session=session)

        assert np.allclose(session.run(a(x)), expected, atol=0.01)
        assert np.array_equal(
            session.run(a(x)), session.run(a.forward(ag.Constant(x)))
        )

    @pytest.mark.parametrize(
        "x, expected", zip(TEST_CASES, backward_expected), ids=TEST_IDS
    )
    def test_gradient(self, session, x, expected):
        a = activation.Softplus(session=session)

        x_node = ag.Constant(x)
        y = a(x_node)
        _ = session.run(y)

        assert np.allclose(session.gradients(y, [x_node]), expected, atol=0.01)
