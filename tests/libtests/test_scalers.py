import numpy as np
import pytest

from api.lib.preprocessing import scalers


def test_standard_scaler():
    sc = scalers.StandardScaler()
    x = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])

    sc.fit(x)
    assert sc.mu is not None and sc.sigma is not None

    with pytest.raises(TypeError):
        scalers.StandardScaler().transform(x)

    transformed = sc.transform(x)
    tr_mean = np.mean(transformed, axis=0)
    tr_std = np.std(transformed, axis=0)
    assert np.array_equal(tr_mean, np.zeros(tr_mean.shape))
    assert np.array_equal(tr_std, np.ones(tr_std.shape))

    fit_transformed = scalers.StandardScaler().fit_transform(x)
    assert np.array_equal(fit_transformed, transformed)


def test_minmax_scaler():
    sc = scalers.MinMaxScaler()
    x = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])

    sc.fit(x)
    assert sc.min_ is not None and sc.max_ is not None

    with pytest.raises(TypeError):
        scalers.MinMaxScaler().transform(x)

    transformed = sc.transform(x)
    assert np.all(np.logical_and((transformed >= 0), (transformed <= 1)))

    fit_transformed = scalers.MinMaxScaler().fit_transform(x)
    assert np.array_equal(transformed, fit_transformed)

    fr_range = (-10, 10)
    sc = scalers.MinMaxScaler(feature_range=fr_range)
    transformed = sc.fit_transform(x)
    assert np.all(
        np.logical_and(
            (transformed >= fr_range[0]),
            (transformed <= fr_range[1])
        )
    )

