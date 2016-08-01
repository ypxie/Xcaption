import backend.export as T

from  backend.export import npwrapper
from Core.utils_func import *
import numpy as np
from utils import activations, initializations, regularizers

# dropout in theano
def dropout_layer(state_before, rng =None,dropoutrate=0.5):
    """
    tensor switch is like an if statement that checks the
    value of the theano shared variable (use_noise), before
    either dropping out the state_before tensor or
    computing the appropriate activation. During training/testing
    use_noise is toggled on and off.
    """
    if 0. < dropoutrate < 1.:
        retain_p = 1. - dropoutrate
        B = T.binomial(shape = state_before.shape, p=retain_p, n=1,dtype=state_before.dtype,rng = rng) * (1. / retain_p)
        proj = T.in_train_phase(state_before * B, state_before)
    return proj
    # proj = T.switch(use_noise,
    #                      state_before *
    #                      T.binomial(shape = state_before.shape, p=p, n=1,dtype=state_before.dtype,rng = rng),
    #                      state_before * p)
    # return proj

    
# ----------------------embeding layer----------------------    
def init_embeding(options, params, prefix='embeding',input_dim=None,
                  output_dim=None, init='norm_weight',trainable=True):
    '''
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
def embeding_layer(tparams, x, options, prefix='embeding',dropoutrate=None,
                    specifier=None,filled_value=0., **kwargs):
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
    if 0. < dropoutrate < 1.:
        retain_p = 1. - dropoutrate
        B = T.binomial(shape=(W.shape[0],), p=retain_p) * (1. / retain_p)
        B = T.expand_dims(B)
        W = T.in_train_phase(W * B, W)
    
    # for the first word (which is coded with -1), emb should be all zero
    #emb = tensor.switch(x[:,None] < 0, tensor.alloc(0., 1, tparams['Wemb'].shape[1]),
    #                    tparams['Wemb'][x])
                     
    if specifier is not None:
        filled_shape = [1 for _ in range(T.ndim(x))] + [W.shape[-1]]
        if T.ndim(x) == 1:
            comp = x[:,None]
        elif T.ndim(x) == 2:
            comp = x[:,:,None]
        elif T.ndim(x) == 3:
            comp = x[:,:,:,None]
        else:
            raise Exception('dimention {} is not supported yet!'.format(T.ndim(x))  )
        emb = T.switch(T.equal(comp, specifier), T.alloc(filled_value,*filled_shape),
                            W[x])
    else:
        emb = W[x]
    emb._keras_shape = tuple(x._keras_shape) + (output_dim,)
    return emb

def Embedding(tparams, options, x,  params = None, prefix='embeding',input_dim=None,
              output_dim=None, init='norm_weight',trainable=True, dropoutrate=None,
              specifier=None,filled_value=0., belonging_Module=None,**kwargs):
    '''
    params > tparams > empty
    if params covers all the weights_keys. use params to update tparams.
    '''
    module_identifier = 'layer_' + prefix
    init_LayerInfo(options, name = module_identifier)
    if not belonging_Module:
        belonging_Module = options['belonging_Module'] if hasattr(options,'belonging_Module') else None
    else:
        belonging_Module = belonging_Module
           
    input_shape = x._keras_shape
    tmp_param = OrderedDict()
    tmp_param = init_embeding(options, tmp_param, prefix=prefix,input_dim=input_dim,
                              output_dim=output_dim, init=init,trainable=trainable)
    update_or_init_params(tparams, params, tmp_params=tmp_params)
    
    output = embeding_layer(tparams,options, x, dropoutrate=dropoutrate,
                            specifier=specifier,filled_value=filled_value,**kwargs)

    update_father_module(options,belonging_Module, module_identifier)
    return output

# ----------------------fully connected layer----------------------    
# feedforward layer: affine transformation + point-wise nonlinearity
def init_fflayer(options, params, prefix='ff', nin=None, 
                 nout=None,init='norm_weight',trainable=True,**kwargs):
    if nin is None:
        nin = options['dim_proj']
    if nout is None:
        nout = options['dim_proj']

    init = initializations.get(init)    
    params[get_name(prefix, 'W')] = npwrapper(init((nin, nout),scale=0.01,symbolic=False), trainable=trainable) 
    params[get_name(prefix, 'b')] = npwrapper(initializations.get('zero')((nout,),symbolic=False).astype('float32'), trainable=trainable) 

    return params

def fflayer(tparams, x, options, prefix='ff', activation='tanh', **kwargs):
    activation_func = activations.get(activation) 
    return activation_func(T.dot(x, tparams[get_name(prefix,'W')])+tparams[get_name(prefix,'b')])

def Dense(tparams, x,  options, params = None, prefix='ff', nin=None, nout=None,
          init='norm_weight',trainable=True, activation='tanh',belonging_Module=None,**kwargs):
    '''
    params > tparams > empty
    if params covers all the weights_keys. use params to update tparams.
    '''
    module_identifier = 'layer_' + prefix
    init_LayerInfo(options, name = module_identifier)
    if not belonging_Module:
        belonging_Module = options['belonging_Module'] if hasattr(options,'belonging_Module') else None
    else:
        belonging_Module = belonging_Module
        
    tmp_param = OrderedDict()
    tmp_param = init_fflayer(options, tmp_param, prefix=prefix, nin=nin, 
                             nout=nout,init=init,trainable=trainable)
    update_or_init_params(tparams, params, tmp_params=tmp_params)
    
    output = fflayer(tparams, x,options, prefix=prefix, activation= activation, **kwargs)

    update_father_module(options,belonging_Module, module_identifier)

    return output