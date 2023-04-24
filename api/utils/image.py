"""Utilities to work with image data."""
from io import BytesIO

import numpy as np
from PIL import Image

ALLOWED_IMAGE_EXT = ["jpg", "jpeg", "png"]
ALLOWED_IMAGE_HEIGHT = 28
ALLOWED_IMAGE_WIDTH = 28

IMAGE_SIZE = ALLOWED_IMAGE_HEIGHT * ALLOWED_IMAGE_WIDTH  # pixels


def preprocess_image(image):
    """Preprocess image to fit model input."""

    image_size = image.size[0] * image.size[1]
    if image_size != IMAGE_SIZE:
        raise ValueError(
            f"Image should be of size: {IMAGE_SIZE}, received: {image_size}"
        )

    # convert image into flat numpy array of shape (1, IMAGE_SIZE)
    image = np.array(image.getdata()).reshape((1, IMAGE_SIZE))

    # normalize inputs from 0-255 to 0-1
    image = image / 255

    return image


def read_imagefile(file):
    """Read the image into memory and preprocess it."""
    return preprocess_image(Image.open(BytesIO(file)).convert("L")).tolist()
