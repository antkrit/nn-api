"""Contains various useful classes and functions"""


class dotdict(dict):  # pylint: disable=C0103
    """dot.notation access to dictionary attributes."""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
