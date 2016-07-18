# layers: 'name': ('parameter initializer', 'fprop')
import numpy as np 

import backend.export as T
from collections import OrderedDict
from backend.export import npwrapper

def wta(X):
    M = T.max(X, axis=-1, keepdims=True)
    R =T.switch(T.equal(X, M), X, 0.)
    return R
def renorm(x):
    return x / (x.sum(axis=1, keepdims=True))

def circulant(leng, n_shifts):
    """
    I confess, I'm actually proud of this hack. I hope you enjoy!
    This will generate a tensor with `n_shifts` of rotated versions the
    identity matrix. When this tensor is multiplied by a vector
    the result are `n_shifts` shifted versions of that vector. Since
    everything is done with inner products, everything is differentiable.
    Paramters:
    ----------
    leng: int > 0, number of memory locations, can be tensor variable
    n_shifts: int > 0, number of allowed shifts (if 1, no shift)
    Returns:
    --------
    shift operation, a tensor with dimensions (n_shifts, leng, leng)
    """
    #eye = np.eye(leng)
    #shifts = range(n_shifts//2, -n_shifts//2, -1)
    #C = np.asarray([np.roll(eye, s, axis=1) for s in shifts])
    #return theano.shared(C.astype(theano.config.floatX))
    eye = T.eye(leng)
    shifts = range(n_shifts//2, -n_shifts//2, -1)
    C = T.stack([T.roll(eye, s, axis=1) for s in shifts], axis = 0)
    return C  
    
def cosine_distance(M, k):
    #M = theano.printing.Print('[Debug] M shape is: ', attrs=("shape",))(M)
    dot = (M * k[:, None, :]).sum(axis=-1)
    nM = T.sqrt((M**2).sum(axis=-1))
    nk = T.sqrt((k**2).sum(axis=-1, keepdims=True))
    return dot / (nM * nk)  
    
def quodra_distance(M,W, k):
    #M = theano.printing.Print('[Debug] M shape is: ', attrs=("shape",))(M)
    
    dot = (T.dot(M,W) * k[:, None, :]).sum(axis=-1)
    nM = T.sqrt((M**2).sum(axis=-1))
    nk = T.sqrt((k**2).sum(axis=-1, keepdims=True))
    return dot / (nM * nk)  
    
#def update_controller(self, inp, h_tm1, M):
#    """We have to update the inner RNN inside the NTM, this
#    is the function to do it. Pretty much copy+pasta from Keras
#    """
#    x = T.concatenate([inp, M], axis=-1)
#    #1 is for gru, 2 is for lstm
#    if len(h_tm1) in [1,2]:
#        if hasattr(self.rnn,"get_constants"):
#            BW,BU = self.rnn.get_constants(x)
#            h_tm1 += (BW,BU)
#    # update state
#    _, h = self.rnn.step(x, h_tm1)
#
#    return h

# some utilities
# def ortho_weight(ndim):
#     """
#     Random orthogonal weights
#     Used by norm_weights(below), in which case, we
#     are ensuring that the rows are orthogonal
#     (i.e W = U \Sigma V, U has the same
#     # of rows, V has the same # of cols)
#     """
#     if type(ndim) == int:
#         shape = (ndim, ndim)
#     else:
#         shape = tuple(ndim)
#     W = np.random.randn(*shape)
    
#     u, _, v = np.linalg.svd(W, full_matrices=False)
#     # pick the one with the correct shape
#     q = u if u.shape == shape else v
#     q = q.reshape(shape)
#     return q.astype('float32') 
    
# def zero_weight(shape):
#     return np.zeros(shape).astype('float32')
    
# def norm_weight(nin,nout=None, scale=0.01, ortho=True):
#     """
#     Random weights drawn from a Gaussian
#     """
#     if nout is None:
#         nout = nin
#     if nout == nin and ortho:
#         W = ortho_weight(nin)
#     else:
#         W = scale * np.random.randn(nin, nout)
#     return W.astype('float32')

# # some useful shorthands
# def tanh(x):
#     return T.tanh(x)

# def rectifier(x):
#     return T.maximum(0., x)

# def linear(x):
#     return x

'''
Theano uses shared variables for parameters, so to
make this code more portable, these two functions
push and pull variables between a shared
variable dictionary and a regular np
dictionary
'''
# push parameters to Theano shared variables
def zipp(params, tparams):
    for kk, vv in params.iteritems():
        #tparams[kk].set_value(vv)
        T.set_value(tparams[kk], vv)

# pull parameters from Theano shared variables
def unzip(zipped):
    new_params = OrderedDict()
    for kk, vv in zipped.iteritems():
        new_params[kk] = T.get_value(vv)
    return new_params

# get the list of parameters: Note that tparams must be OrderedDict
def itemlist(tparams):
    return [vv for _, vv in tparams.iteritems()]

# make prefix-appended name
def get_name(pp, name):
    return '%s_%s' % (pp, name)

# initialize Theano shared variables according to the initial parameters
def init_tparams(params):
    tparams = OrderedDict()
    for kk, _ in params.iteritems():
        tparams[kk] = T.variable(params[kk], name=kk)
        tparams[kk].trainable = params[kk].trainable
    return tparams

# load parameters
def load_params(path, params):
    pp = np.load(path)
    for kk, vv in params.iteritems():
        if kk not in pp:
            raise Warning('%s is not in the archive' % kk)
        if hasattr(params[kk], 'trainable'):
           params[kk] = npwrapper(pp[kk], trainable=params[kk].trainable)
        else:
           params[kk] = npwrapper(pp[kk], trainable=True)
    return params
