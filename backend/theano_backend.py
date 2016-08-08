from theano.tensor import *
from backend.keras_backend.theano_backend import *
from backend.keras_backend.common import *

from theano import scan, shared, function
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
_FLOATX = 'float32'
floatX = _FLOATX
try:
    from theano.tensor.nnet.nnet import softsign as T_softsign
except ImportError:
    from theano.sandbox.softsign import softsign as T_softsign

def assign_subtensor(dest, source, dest_slice):
    if hasattr(dest, '_keras_shape'):
       ks = dest._keras_shape
    dest = T.set_subtensor(dest[dest_slice], source)    
    dest._keras_shape = ks
    return dest    
    
def isnan(x):
    return x==None

def reshape(x, shape):
    '''
    For this function, it is not possible to reliablly infer it's keras shape    
    '''
    output = T.reshape(x, shape)      
    return output


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

def binomial(shape=None, p=0.0, n=1,dtype=_FLOATX, rng=None):
    if rng is None:
        seed = np.random.randint(1, 10e6)
        rng = RandomStreams(seed=seed)
    if shape is None:
        shape = p.shape
    return rng.binomial(size = shape, p=p, n= 1, dtype=dtype)

def multinomial(shape=None, p=0.0, n =1, dtype=_FLOATX, rng=None):
    if rng is None:
        seed = np.random.randint(1, 10e6)
        rng = RandomStreams(seed=seed)
    if shape is None:
        shape = p.shape
    return rng.binomial(n=n, p=p,size= shape,dtype=dtype)
   
   
