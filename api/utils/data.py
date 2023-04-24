"""Contains utilities to work with data."""
import numpy as np


def decode_mnist_model_output(output, n_entries=1):
    """Converting the output of the MNIST model to a user-friendly format.

    :param output: model output
    :param n_entries: number of the highest elements to return
    :return: dict, index-value pairs
    """
    output = np.ravel(output)
    sorted_ = np.argsort(output)[::-1].tolist()

    return dict(zip(sorted_[:n_entries], output[sorted_][:n_entries]))
