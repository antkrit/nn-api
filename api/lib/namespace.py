"""Defines containers with available names that are used to identify and refer
to objects of various kinds.
"""
from api.lib import autograd
from api.lib.utils import Container
from api.lib.activation import *
from api.lib.loss import *
from api.lib.optimizers import *
from api.lib.exception import *
from api.lib.bases import *
from api.lib.preprocessing.initializers import *


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
bases = Container(
    name='bases',

    Loss=BaseLoss,
    Layer=BaseLayer,
    Activation=BaseActivation,
    Initializer=BaseInitializer,
    Optimizer=BaseOptimizer,
    Scaler=BaseScaler,
)
nodes = autograd.utils.NODES_CONTAINER
