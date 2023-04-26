import json
import logging
import sys
import traceback

import pytest
from PIL import Image

from api.utils.data import decode_mnist_model_output
from api.utils.image import IMAGE_SIZE, preprocess_image
from api.utils.logging import JSONLogFormatter
from api.v1.schemas import BaseJSONLog
from tests.utils import create_in_memory_image


def test_preprocess_image():
    image = Image.open(create_in_memory_image(size=(25, 25)))

    with pytest.raises(ValueError):
        preprocess_image(image)

    image = Image.open(create_in_memory_image(size=(28, 28)))
    image = preprocess_image(image)
    assert image.shape == (1, IMAGE_SIZE)

    image = Image.open(create_in_memory_image(size=(7, 112)))
    image = preprocess_image(image)
    assert image.shape == (1, IMAGE_SIZE)

    image = Image.open(create_in_memory_image(size=(1, 784)))
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


def test_json_log_formatter():
    a = JSONLogFormatter()

    msg = "test"
    log_record = logging.LogRecord(
        "test", 10, "tests/", 1, msg, args=(), exc_info=None
    )

    log_string = json.loads(a.format(log_record))
    # check if log_record match BaseJSONLog scheme
    assert BaseJSONLog(**log_string) is not None

    log_record = logging.LogRecord(
        "test", 10, "tests/", 1, msg, args=(), exc_info=None
    )
    log_record.duration = -1

    log_string = json.loads(a.format(log_record))
    assert BaseJSONLog(**log_string) is not None

    json_log = BaseJSONLog(**log_string).dict()
    assert json_log["duration"] == -1

    try:
        raise ValueError("testing exc_info")
    except ValueError:
        info = sys.exc_info()

    log_record = logging.LogRecord(
        "test", 10, "tests/", 1, msg, args=(), exc_info=info
    )

    log_string = json.loads(a.format(log_record))
    assert BaseJSONLog(**log_string) is not None

    json_log = BaseJSONLog(**log_string).dict()
    assert json_log["exceptions"] == traceback.format_exception(*info)

    log_record = logging.LogRecord(
        "test", 10, "tests/", 1, msg, args=(), exc_info=None
    )
    log_record.exc_text = "exc_text"

    log_string = json.loads(a.format(log_record))
    assert BaseJSONLog(**log_string) is not None

    json_log = BaseJSONLog(**log_string).dict()
    assert json_log["exceptions"] == "exc_text"
