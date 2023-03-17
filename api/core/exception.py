"""Custom exception implementations."""


class ModelIsNotCompiledException(Exception):
    """Raise when model is not compiled."""


class NoGradientException(Exception):
    """Raise for nodes that have no gradients."""
