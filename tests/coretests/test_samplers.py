import numpy as np

from api.core.preprocessing.samplers import train_test_split


def test_train_test_split():
    x_len, percentile = 100, 0.7
    x = np.arange(0, x_len * 2).reshape((x_len, 2))
    y = np.arange(x_len)

    x_train, x_test, y_train, y_test = train_test_split(x, y, split=percentile)

    assert len(x_train) == len(y_train) == int(x_len * percentile)
    assert len(x_test) == len(y_test) == int(x_len * (1 - percentile))

    sid = int(x_len * percentile)
    assert np.array_equal(x_train, x[:sid])
    assert np.array_equal(y_train, y[:sid])
    assert np.array_equal(x_test, x[sid:])
    assert np.array_equal(y_test, y[sid:])

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, split=percentile, shuffle=True, seed=42
    )

    assert not np.array_equal(x_train, x[:sid])
    assert not np.array_equal(y_train, y[:sid])
    assert not np.array_equal(x_test, x[sid:])
    assert not np.array_equal(y_test, y[sid:])

    # test seed
    x_train_42, x_test_42, y_train_42, y_test_42 = train_test_split(
        x, y, split=percentile, shuffle=True, seed=42
    )

    assert np.array_equal(x_train, x_train_42)
    assert np.array_equal(y_train, y_train_42)
    assert np.array_equal(x_test, x_test_42)
    assert np.array_equal(y_test, y_test_42)

    x_train, x_test, y_train, y_test = train_test_split(x, y, split=1)

    assert len(x_train) == len(y_train) == x_len
    assert len(x_test) == len(y_test) == 0
