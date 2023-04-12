"""Defines containers with available names that are used to identify and refer
to objects of various kinds.
"""
from api.core.data import Container
from api.core.autograd import (
    Node, Constant, Variable, Placeholder, Operation,
    UnaryOperation, BinaryOperation
)
from api.core.activation import (
    Sigmoid, Tanh, Swish, Softmax, Softplus, ReLU, ELU
)
from api.core.exception import ModelIsNotCompiledException, NoGradientException
from api.core.loss import MSE, MAE, MBE, BCE, Huber, Hinge, LHL, CCE, KLD
from api.core.optimizers import (
    GradientDescent, Adagrad, Adadelta, Adam, Adamax, RMSProp
)
from api.core.preprocessing.initializers import (
    zeros, ones, random_normal, random_uniform, he_normal, he_uniform,
    xavier_normal, xavier_uniform, lecun_normal, lecun_uniform
)


activations = Container(
    name='activations',

    sigmoid=Sigmoid,
    tanh=Tanh,
    swish=Swish,
    softmax=Softmax,
    softplus=Softplus,
    relu=ReLU,
    elu=ELU,
)
losses = Container(
    name='losses',

    mean_squared_error=MSE,
    mean_absolute_error=MAE,
    mean_bias_error=MBE,
    huber_loss=Huber,
    likelihood_loss=LHL,
    binary_cross_entropy=BCE,
    hinge_loss=Hinge,
    categorical_cross_entropy=CCE,
    kullback_leibler_divergence=KLD,
)
initializers = Container(
    name='initializers',

    zeros=zeros,
    ones=ones,
    random_normal=random_normal,
    random_uniform=random_uniform,
    he_normal=he_normal,
    he_uniform=he_uniform,
    xavier_normal=xavier_normal,
    xavier_uniform=xavier_uniform,
    lecun_normal=lecun_normal,
    lecun_uniform=lecun_uniform,
)
optimizers = Container(
    name='optimizers',

    gradient_descent=GradientDescent,
    adagrad=Adagrad,
    adadelta=Adadelta,
    rmsprop=RMSProp,
    adam=Adam,
    adamax=Adamax,
)
exceptions = Container(
    name='exceptions',

    ModelIsNotCompiled=ModelIsNotCompiledException,
    NoGradient=NoGradientException
)
nodes = Container(
    name='nodes',

    node=Node,
    constant=Constant,
    variable=Variable,
    placeholder=Placeholder,
    operation=Operation,
    unary_operation=UnaryOperation,
    binary_operation=BinaryOperation
)
