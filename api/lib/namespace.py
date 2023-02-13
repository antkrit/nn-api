"""Defines containers with available names that are used to identify and refer to objects of various kinds."""
from collections.abc import MutableMapping
from api.lib.activation import *
from api.lib.preprocessing.initializers import *
from api.lib.loss import *
from api.lib.optimizers import *
from api.lib.exception import *
from api.lib.bases import *


__all__ = ('activations', 'initializers', 'losses', 'optimizers')


class Container(MutableMapping):
    """Base dict-like container class.

    container = BaseContainer(name=..., **kwargs)
    to get an object - use dict syntax, e.g. container['obj_name']
    to get the compiled instance - use __call__ method,
    e.g. container('obj_name', *args, **kwargs)
    """

    def __init__(self, name, *args, **kwargs):
        """Constructor method."""
        self.name = name
        self.store = dict()
        self.update(dict(*args, **kwargs))

    def __getitem__(self, key):
        return self.store[key]

    def __getattr__(self, item):
        return self.__getitem__(item)

    def __setitem__(self, key, value):
        self.store[key] = value

    def __delitem__(self, key):
        del self.store[key]

    def __call__(self, obj_name, compiled=False, *args, **kwargs):
        obj = self.__getitem__(key=obj_name)
        if callable(obj) and compiled:
            return obj(*args, **kwargs)
        return obj

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)

    def __repr__(self):
        return f'Container-{self.name}({self.store.items()})'


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
optimizers = Container(
    name='optimizers',

    gradient_descent=GradientDescent,
)
exceptions = Container(
    name='exceptions',

    ModelIsNotCompiled=ModelIsNotCompiledException,
)
bases = Container(
    name='bases',

    Layer=BaseLayer,
    Activation=BaseActivation,
    Loss=BaseLoss,
    Initializer=BaseInitializer,
    Optimizer=BaseOptimizer,
    Scaler=BaseScaler,
)
