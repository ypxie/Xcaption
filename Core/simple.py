import backend.export as T

from  backend.export import npwrapper
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

    params['Wemb'] = norm_weight(options['n_words'], options['dim_word'])

#embeding layer
def init_embeding(options, params, prefix='embeding',input_dim=None,
                  output_dim=None, init='normal',trainable=True):
''''
    Turn positive integers (indexes) into dense vectors of fixed size.
    eg. [[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]
# Arguments: copy from keras 
      input_dim: int > 0. Size of the vocabulary, ie.
          1 + maximum integer index occurring in the input data.
      output_dim: int >= 0. Dimension of the dense embedding.
      init: name of initialization function for the weights
          of the layer (see: [initializations](../initializations.md)),
          or alternatively, Theano function to use for weights initialization.
          This parameter is only relevant if you don't pass a `weights` argument.
      weights: list of Numpy arrays to set as initial weights.
          The list should have 1 element, of shape `(input_dim, output_dim)`.
      W_regularizer: instance of the [regularizers](../regularizers.md) module
        (eg. L1 or L2 regularization), applied to the embedding matrix.
      W_constraint: instance of the [constraints](../constraints.md) module
          (eg. maxnorm, nonneg), applied to the embedding matrix.
      mask_zero: Whether or not the input value 0 is a special "padding"
          value that should be masked out.
          This is useful for [recurrent layers](recurrent.md) which may take
          variable length input. If this is `True` then all subsequent layers
          in the model need to support masking or an exception will be raised.
          If mask_zero is set to True, as a consequence, index 0 cannot be
          used in the vocabulary (input_dim should equal |vocabulary| + 2).
      input_length: Length of input sequences, when it is constant.
          This argument is required if you are going to connect
          `Flatten` then `Dense` layers upstream
          (without it, the shape of the dense outputs cannot be computed).
      dropout: float between 0 and 1. Fraction of the embeddings to drop.
'''
    init = initializations.get(init)
    params[get_name(prefix, 'W')] = npwrapper(init(shape=(input_dim,output_dim), symbolic=False),
                                   trainable = trainable)
                                
    return params
def embeding_layer(tparams, x, options, prefix='embeding',dropout=None,
                    specifier=-1,filled_value=0., **kwargs):
    '''
    Gather weights from W based on index of x. The output should be one dimension
    higher than x, eg, if W = (100,128), x=(10,10,10), output will be (10,10,10,128).
    
    Parameters:
    -----------
       tparams, 
       x : the index of the embeding axis, can be any shape.
       specifier: The special index that you want to specifier a specified value,
       filled_value: The special value to filled based on specifier.
    '''
    W = tparams[get_name(prefix,'W')]
    if 0. < dropout < 1.:
        retain_p = 1. - self.dropout
        B = T.binomial(shape=(self.input_dim,), p=retain_p) * (1. / retain_p)
        B = T.expand_dims(B)
        W = T.in_train_phase(self.W * B, self.W)
    else:
        W = self.W
    filled_shape = [1 for _ in range(len(x.shape))] + [W.shape[-1]]
    emb = T.switch(x[...,None] == specifier, T.alloc(filled_value,*filled_shape),
                        W[x])
    return emb
    
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

