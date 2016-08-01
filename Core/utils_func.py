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
    return dot / (nM * nk + 0.00000001)  
    
def quodra_distance(M,W, k):
    #M = theano.printing.Print('[Debug] M shape is: ', attrs=("shape",))(M)
    
    dot = (T.dot(M,W) * k[:, None, :]).sum(axis=-1)
    nM = T.sqrt((M**2).sum(axis=-1))
    nk = T.sqrt((k**2).sum(axis=-1, keepdims=True))
    return dot / (nM * nk + 0.000000001)  
    
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
def update_tparams(tparams=None,params=None):
    if not tparams:
        tparams = OrderedDict()
    for kk, _ in params.iteritems():
        tparams[kk] = T.variable(params[kk], name=kk)
        tparams[kk].trainable = params[kk].trainable
    return tparams

def init_tparams(params):
    return update_tparams(tparams = None, params=params)
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

def update_or_init_params(tparams=None, params=None, tmp_params=None):
    tmp_params_keys = tmp_params.keys()
    if set(tmp_params_keys).issubset(params.keys()):
        # if params is given, we update tparams from params
        tparams = update_tparams(tparams, params) 
    else:
        # if params does not has all the keys, we need to initliaze it    
        if set(tmp_params_keys).issubset(tparams.keys()):
            # we don't need to do anything, just use tparams
            pass
        else:
            # if tparams does not has all the keys, it means we need to initlize params and then tparams
            tparams = update_tparams(tparams, tmp_params)
            for k,v in tmp_params.iteritems():
                params[k] = v
    return tparams, params

class obj(object):
    pass

def init_LayerInfo(options, name):
    thisModule = obj()
    thisModule.trainable_weights = []
    thisModule.non_trainable_weights = []
    thisModule.regularizers = []
    thisModule.constraints = []
    options[name] = thisModule
    return options

def init_ModuleInfo(options, name):
    thisModule = obj()
    thisModule.trainable_weights = []
    thisModule.regularizers = []
    thisModule.constraints = []
    options[name] = thisModule
    return options

def update_father_module(options,belonging_Module, module_identifier):
    '''
    we need to update the father module which contains this layer based on the information
    in this layer
    '''
    if not belonging_Module:
        # means we need to update the father module info mation here
        if not belonging_Module in options:
            raise Warning('father module: {m} not initialized before calling: {f}'.
                         format(m=belonging_Module,f=module_identifier))
            init_ModuleInfo(options, belonging_Module)
        thisInfo = options['module_identifier']

        belongingModuleInfo = options['belonging_Module']
        belongingModuleInfo.trainable_weights += thisInfo.trainable_weights
        belongingModuleInfo.non_trainable_weights += thisInfo.non_trainable_weights
        belongingModuleInfo.regularizers += thisInfo.regularizers
        belongingModuleInfo.constraints  += thisInfo.constraints
        return options 

def get_subdict_prefix(tparams, prefixlist=None):
    if not isinstance(prefixlist, list):
        prefixlist = [prefixlist]
    prefixlist = tuple(prefixlist)
    subParams = OrderedDict()

    for k in tparams.keys():
        if k.startswith(prefixlist):
            subParams[k] = tparams[k]
    return subParams

def load_keras_model(tparams, params, keras_model_path):
    '''
    Load all layer weights from a HDF5 save file.
    '''
    import h5py
    f = h5py.File(keras_model_path, mode='r')
    
    if hasattr(self, 'flattened_layers'):
        # support for legacy Sequential/Merge behavior
        flattened_layers = self.flattened_layers
    else:
        flattened_layers = self.layers

    if 'nb_layers' in f.attrs:
        # legacy format
        nb_layers = f.attrs['nb_layers']
        if nb_layers != len(flattened_layers):
            raise Exception('You are trying to load a weight file '
                            'containing ' + str(nb_layers) +
                            ' layers into a model with ' +
                            str(len(flattened_layers)) + '.')

        for k in range(nb_layers):
            g = f['layer_{}'.format(k)]
            weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
            flattened_layers[k].set_weights(weights)
    else:
        # new file format
        layer_names = [n.decode('utf8') for n in f.attrs['layer_names']]
        if len(layer_names) != len(flattened_layers):
            import warnings
            warnings.warn('You are trying to load a weight file '
                            'containing ' + str(len(layer_names)) +
                            ' layers into a model with ' +
                            str(len(flattened_layers)) + ' layers.')

            #raise Exception('You are trying to load a weight file '
            #                'containing ' + str(len(layer_names)) +
            #                ' layers into a model with ' +
            #                str(len(flattened_layers)) + ' layers.')

        # we batch weight value assignments in a single backend call
        # which provides a speedup in TensorFlow.
        weight_value_tuples = []
        para_ind = -1
        non_empty_layers = []
        for pind,flayer in enumerate(flattened_layers):
            symbolic_weights = flayer.trainable_weights + flayer.non_trainable_weights
            if len(symbolic_weights) != 0:
                non_empty_layers.append(flayer)
        
        for k, name in enumerate(layer_names):
            g = f[name]
            weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
            if len(weight_names):
                para_ind = para_ind + 1
                weight_values = [g[weight_name] for weight_name in weight_names]
                layer = non_empty_layers[para_ind]
                symbolic_weights = layer.trainable_weights + layer.non_trainable_weights
                if len(weight_values) != len(symbolic_weights):
                    raise Exception('Layer #' + str(para_ind) +
                                    ' (named "' + layer.name +
                                    '" in the current model) was found to '
                                    'correspond to layer ' + name +
                                    ' in the save file. '
                                    'However the new layer ' + layer.name +
                                    ' expects ' + str(len(symbolic_weights)) +
                                    ' weights, but the saved weights have ' +
                                    str(len(weight_values)) +
                                    ' elements.')
                weight_value_tuples += zip(symbolic_weights, weight_values)
        K.batch_set_value(weight_value_tuples)
    f.close()