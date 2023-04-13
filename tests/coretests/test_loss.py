import numpy as np
import pytest

import api.core.autograd as ag
from api.core.loss import *


def test_representation(mocker):
    mocker.patch.object(BaseLoss, "forward", lambda *args, **kwargs: 10.0)

    bl = BaseLoss(name="some_name")
    _ = bl(None, None)
    assert str(bl) == f"{bl.name}: {10.0}"


@pytest.mark.parametrize(
    "y_true, y_pred, expected",
    [
        ([1, -1, 1], [1, -1, 1], 0),
        (0, -1, 1),
        ([1, -1, 1], [0, 0, 0], 1),
        ([[3, 3, 3], [2, 2, 2]], [[2, 2, 2], [3, 3, 3]], 1),
    ],
    ids=["error=0", "scalar", "vector", "matrix"],
)
class TestRegressionLosses:
    @pytest.mark.parametrize("root", [True, False], ids=["rmse", "mse"])
    def test_mse_rmse(self, session, y_true, y_pred, expected, root):
        lss = MSE(session=session, root=root)

        assert np.array_equal(
            session.run(lss(y_pred, y_true)),
            np.sqrt(expected) if root else expected,
        )
        assert np.array_equal(
            session.run(
                lss(
                    y_pred,
                    y_true,
                )
            ),
            session.run(lss.forward(ag.Constant(y_pred), ag.Constant(y_true))),
        )

    def test_mae(self, session, y_true, y_pred, expected):
        lss = MAE()

        assert np.allclose(session.run(lss(y_pred, y_true)), expected)
        assert np.array_equal(
            session.run(lss(y_pred, y_true)),
            session.run(lss.forward(ag.Constant(y_pred), ag.Constant(y_true))),
        )

    def test_mbe(self, session, y_true, y_pred, expected):
        lss = MBE()

        assert np.allclose(
            session.run(lss(y_pred, y_true)),
            np.mean(np.asarray(y_pred) - np.asarray(y_true)),
        )
        assert np.array_equal(
            session.run(lss(y_pred, y_true)),
            session.run(lss.forward(ag.Constant(y_pred), ag.Constant(y_true))),
        )

    @pytest.mark.parametrize("delta", [1, 2], ids=["delta=1", "delta=2"])
    def test_huber(self, session, y_true, y_pred, expected, delta):
        lss = Huber(session=session, delta=delta)

        assert np.array_equal(
            session.run(lss(y_pred, y_true)),
            expected / 2,  # note: div by 2 is just hardcode to fit the expected
        )
        assert np.array_equal(
            session.run(lss(y_pred, y_true)),
            session.run(lss.forward(ag.Constant(y_pred), ag.Constant(y_true))),
        )


class TestBinaryClassificationLosses:
    @pytest.mark.parametrize(
        "y_pred, y_true, expected",
        [
            ([0, 1, 1, 0, 1, 0], [0, 1, 1, 0, 1, 0], 1),
            ([0.3, 0.7, 0.8, 0.5, 0.6, 0.4], [0, 1, 1, 0, 1, 0], 0.65),
        ],
        ids=["error=0", "error!=0"],
    )
    def test_lhl(self, session, y_true, y_pred, expected):
        lss = LHL()

        assert np.allclose(session.run(lss(y_pred, y_true)), expected)
        assert np.array_equal(
            session.run(lss(y_pred, y_true)),
            session.run(lss.forward(ag.Constant(y_pred), ag.Constant(y_true))),
        )

    @pytest.mark.parametrize(
        "y_pred, y_true, expected",
        [
            ([0, 1, 1, 0, 1, 0], [0, 1, 1, 0, 1, 0], 0),
            ([0.3, 0.7, 0.8, 0.5, 0.6, 0.4], [0, 1, 1, 0, 1, 0], 0.191906),
        ],
        ids=["error=0", "error!=0"],
    )
    def test_bce(self, session, y_pred, y_true, expected):
        lss = BCE()

        assert np.allclose(session.run(lss(y_pred, y_true)), expected)
        assert np.array_equal(
            session.run(lss(y_pred, y_true)),
            session.run(lss.forward(ag.Constant(y_pred), ag.Constant(y_true))),
        )

    @pytest.mark.parametrize(
        "y_pred, y_true, expected",
        [
            ([-1, -1, -1], [-1, -1, -1], 0),
            (
                [1, -1, 1, 1, -1, -1, -1, -1, -1, -1, 1, 1],
                [-1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 1, 1],
                0.5,
            ),
        ],
        ids=["error=0", "error!=0"],
    )
    def test_hinge(self, session, y_true, y_pred, expected):
        lss = Hinge()

        assert np.allclose(session.run(lss(y_pred, y_true)), expected)
        assert np.array_equal(
            session.run(lss(y_pred, y_true)),
            session.run(lss.forward(ag.Constant(y_pred), ag.Constant(y_true))),
        )


@pytest.mark.parametrize(
    "y_true, y_pred, expected",
    [
        ([[1, 0], [0, 1], [0, 1]], [[1, 0], [0, 1], [0, 1]], 0),
        (
            [[1, 0], [0, 1], [0, 1], [1, 0]],
            [[0.9, 0.1], [0.1, 0.9], [0.2, 0.8], [0.35, 0.65]],
            0.080544,
        ),
    ],
    ids=["error=0", "error!=0"],
)
class TestMultinomialClassificationLosses:
    def test_cce(self, session, y_true, y_pred, expected):
        lss = CCE()

        assert np.allclose(session.run(lss(y_pred, y_true)), expected)
        assert np.array_equal(
            session.run(lss(y_pred, y_true)),
            session.run(lss.forward(ag.Constant(y_pred), ag.Constant(y_true))),
        )

    def test_kld(self, session, y_true, y_pred, expected):
        lss = KLD()

        assert np.allclose(session.run(lss(y_pred, y_true)), expected)
        assert np.array_equal(
            session.run(lss(y_pred, y_true)),
            session.run(lss.forward(ag.Constant(y_pred), ag.Constant(y_true))),
        )


class TestOtherLosses:
    pass
