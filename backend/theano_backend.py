from backend.keras_backend.theano_backend import *
from theano.tensor import *
from theano import scan, shared, function
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
_FLOATX = 'float32'
floatX = _FLOATX
try:
    from theano.tensor.nnet.nnet import softsign as T_softsign
except ImportError:
    from theano.sandbox.softsign import softsign as T_softsign

def assign_subtensor(dest, source, slice_dest):
    dest = T.set_subtensor(dest[slice_dest], source)    
    return dest    
    
    
def isnan(x):
    return x==None

def reshape(x, shape):
    return T.reshape(x, shape)


def sigmoid(x):
    return T.nnet.sigmoid(x)

def hard_sigmoid(x):
    return T.nnet.hard_sigmoid(x)

def tanh(x):
    return T.tanh(x)    
def relu(x, alpha=0., max_value=None):
    assert hasattr(T.nnet, 'relu'), ('It looks like like your version of '
                                     'Theano is out of date. '
                                     'Install the latest version with:\n'
                                     'pip install git+git://github.com/Theano/Theano.git --upgrade --no-deps')
    x = T.nnet.relu(x, alpha)
    if max_value is not None:
        x = T.minimum(x, max_value)
    return x


def softmax(x):
    return T.nnet.softmax(x)


def softplus(x):
    return T.nnet.softplus(x)


def softsign(x):
    return T_softsign(x)

def normal(shape, mean=0.0, std=1.0, dtype=_FLOATX, rng=None):
    if rng is None:
        seed = np.random.randint(1, 10e6)
        rng = RandomStreams(seed=seed)
    return rng.normal(size=shape, avg=mean, std=std, dtype=dtype)

def uniform(shape, low=0.0, high=1.0, dtype=_FLOATX, rng=None):
    if rng is None:
        seed = np.random.randint(1, 10e6)
        rng = RandomStreams(seed=seed)
    return rng.uniform(size = shape, low=low, high=high, dtype=dtype)

def binomial(shape, p=0.0, n=1,dtype=_FLOATX, rng=None):
    if rng is None:
        seed = np.random.randint(1, 10e6)
        rng = RandomStreams(seed=seed)
    return rng.binomial(size = shape, p=p, n= 1, dtype=dtype)

def multinomial(shape=(), pvals=0.0, n =1, dtype=_FLOATX, rng=None):
    if rng is None:
        seed = np.random.randint(1, 10e6)
        rng = RandomStreams(seed=seed)
    return rng.binomial(n=n, p=pvals,size= shape,dtype=dtype)
   
   
