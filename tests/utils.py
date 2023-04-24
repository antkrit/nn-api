import re
from io import BytesIO

import numpy as np
from PIL import Image


def element_wise_equal(a, b):
    """Check if every array element is equal."""
    if len(a) != len(b):
        raise ValueError(
            f"cannot match arrays with length {len(a)} and {len(b)}"
        )
    return all([np.array_equal(a_i, b[i]) for i, a_i in enumerate(a)])


def check_node_name_format(node):
    """Check node name format."""
    rgx = re.compile(r"^graph-\d+/.*-\d+$")
    return re.search(rgx, node.name) is not None


def create_in_memory_image(size=(50, 50), filename="test", filetype="png"):
    """Create in-memory image"""
    file = BytesIO()

    image = Image.new("L", size=size, color=(255,))
    image.save(file, filetype)

    file.name = f"{filename}.{filetype}"
    file.seek(0)

    return Image.open(file)
