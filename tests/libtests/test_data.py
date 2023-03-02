import numpy as np
from api.lib.data import Dataset


def test_dataset():
    x_train = np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
    y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

    gen = Dataset(x_train, y_train, dim=(1, 2), batch_size=4, shuffle=False)
    assert len(gen) == 1

    gen = Dataset(x_train, y_train, dim=(1, 2), batch_size=100, shuffle=False)
    assert len(gen) == 1

    data = next(gen)
    assert np.array_equal(data[0], x_train)
    assert np.array_equal(data[1], y_train)

    gen = Dataset(x_train, y_train, dim=(1, 2), batch_size=2, shuffle=False)
    assert len(gen) == 2

    batch = next(gen)
    assert len(batch) == 2
    assert np.array_equal(batch[0], np.split(x_train, 2)[0])
    assert np.array_equal(batch[1], np.split(y_train, 2)[0])

    batch = next(gen)
    assert np.array_equal(batch[0], np.split(x_train, 2)[1])
    assert np.array_equal(batch[1], np.split(y_train, 2)[1])

    batch = next(gen)
    assert np.array_equal(batch[0], np.split(x_train, 2)[0])
    assert np.array_equal(batch[1], np.split(y_train, 2)[0])

