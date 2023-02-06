"""Defines containers with available names that are used to identify and refer to objects of various kinds."""
from api.lib.bases import BaseContainer
from api.lib.activation import *
from api.lib.preprocessing.initializers import *
from api.lib.loss import *
from api.lib.optimizers import *


__all__ = ('activations', 'initializers', 'losses', 'optimizers')


class ActivationsContainer(BaseContainer):
    """Contains available activations."""
    sigmoid = Sigmoid
    tanh = Tanh
    swish = Swish
    softmax = Softmax
    softplus = Softplus
    relu = ReLU
    elu = ELU


class InitializersContainer(BaseContainer):
    """Contains available initializers."""
    zeros = zeros
    ones = ones
    random_normal = random_normal
    random_uniform = random_uniform
    he_normal = he_normal
    he_uniform = he_uniform
    xavier_normal = xavier_normal
    xavier_uniform = xavier_uniform
    lecun_normal = lecun_normal
    lecun_uniform = lecun_uniform


class LossesContainer(BaseContainer):
    """Contains available losses."""
    mean_squared_error = MSE
    mean_absolute_error = MAE
    mean_bias_error = MBE
    huber_loss = Huber
    likelihood_loss = LHL
    binary_cross_entropy = BCE
    hinge_loss = Hinge
    categorical_cross_entropy = CCE
    kullback_leibler_divergence = KLD


class OptimizersContainer(BaseContainer):
    """Contains available optimizers."""
    gradient_descent = GradientDescent


activations = ActivationsContainer()
initializers = InitializersContainer()
losses = LossesContainer()
optimizers = OptimizersContainer()
