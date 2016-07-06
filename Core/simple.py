import backend.export as T

from backend.export import npwrapper
from Core.utils_func import *
import numpy as np
from utils import activations, initializations, regularizers

# dropout in theano
def dropout_layer(state_before, use_noise, rng =None,p=0.5):
    """
    tensor switch is like an if statement that checks the
    value of the theano shared variable (use_noise), before
    either dropping out the state_before tensor or
    computing the appropriate activation. During training/testing
    use_noise is toggled on and off.
    """
    proj = T.switch(use_noise,
                         state_before *
                         T.binomial(shape = state_before.shape, p=p, n=1,dtype=state_before.dtype,rng = rng),
                         state_before * p)
    return proj

# feedforward layer: affine transformation + point-wise nonlinearity
def init_fflayer(options, params, prefix='ff', nin=None, nout=None,trainable=True,**kwargs):
    if nin is None:
        nin = options['dim_proj']
    if nout is None:
        nout = options['dim_proj']
        
    params[get_name(prefix, 'W')] = npwrapper(norm_weight(nin, nout, scale=0.01), trainable=trainable) 
    params[get_name(prefix, 'b')] = npwrapper(np.zeros((nout,)).astype('float32'), trainable=trainable) 

    return params

def fflayer(tparams, state_below, options, prefix='rconv', activation='relu', **kwargs):
    activation_func = activations.get(activation) 
    return activation_func(T.dot(state_below, tparams[get_name(prefix,'W')])+tparams[get_name(prefix,'b')])

