import pytest
import numpy as np

from api.core.data import Dataset, Container, unpack_x_y


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


def test_base_container():
    init_a_value, next_a_value = 1, 3

    container = Container(name='test', a=init_a_value, b=2)
    assert container['a'] == init_a_value and container['b'] == 2

    container['a'] = next_a_value
    assert container['a'] == next_a_value and container.a == next_a_value

    del container['b']
    assert 'b' not in container
    assert len(container) == 1

    assert container('a') == next_a_value

    class A:
        def __init__(self, a):
            self.a = a

        def __call__(self, *args, **kwargs):
            return self.a

    container['class_A'] = A
    assert container['class_A'] is A
    assert container('class_A') is A
    assert container(A) is A

    instance = container('class_A', compiled=True, a=init_a_value)
    assert instance is not A and isinstance(instance, A)
    assert instance.a == init_a_value


def test_unpack_x_y():
    x, y = unpack_x_y(1)
    assert x == 1 and y is None

    x, y = unpack_x_y([1])
    assert x == 1 and y is None

    x, y = unpack_x_y([1, 2])
    assert x == 1 and y == 2

    with pytest.raises(ValueError):
        _, _ = unpack_x_y([1, 2, 3])
