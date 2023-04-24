import pytest

from api.utils.data import decode_mnist_model_output
from api.utils.image import IMAGE_SIZE, preprocess_image
from tests.utils import create_in_memory_image


def test_preprocess_image():
    image = create_in_memory_image(size=(25, 25))

    with pytest.raises(ValueError):
        preprocess_image(image)

    image = create_in_memory_image(size=(28, 28))
    image = preprocess_image(image)
    assert image.shape == (1, IMAGE_SIZE)

    image = create_in_memory_image(size=(7, 112))
    image = preprocess_image(image)
    assert image.shape == (1, IMAGE_SIZE)

    image = create_in_memory_image(size=(1, 784))
    image = preprocess_image(image)
    assert image.shape == (1, IMAGE_SIZE)


@pytest.mark.parametrize(
    "output",
    ([0.5, 0.4, 0.7], [[0.5, 0.4, 0.7]], [[[0.5, 0.4, 0.7]]]),
    ids=["1d", "2d", "3d"],
)
def test_output_decoding(output):
    decoded = decode_mnist_model_output(output, n_entries=3)
    assert tuple(decoded.keys()) == (2, 0, 1)
    assert tuple(decoded.values()) == (0.7, 0.5, 0.4)

    decoded = decode_mnist_model_output(output, n_entries=1)
    assert tuple(decoded.keys()) == (2,)
    assert tuple(decoded.values()) == (0.7,)
