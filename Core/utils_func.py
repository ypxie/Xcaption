# layers: 'name': ('parameter initializer', 'fprop')
import numpy as np 

import backend.export as T
from collections import OrderedDict
from backend.export import npwrapper
import warnings
def wta(X):
    M = T.max(X, axis=-1, keepdims=True)
    R =T.switch(T.equal(X, M), X, 0.)
    return R
def renorm(x):
    return x / (x.sum(axis=1, keepdims=True)+0.000001)

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
    if tparams is None:
        tparams = OrderedDict()
    for kk, vv in params.iteritems():
        if kk in tparams:
            t_trainable =  getattr(tparams[kk],'trainable', True)
        else:
            t_trainable =  getattr(vv,'trainable', True)
        #print kk
        tparams[kk] = T.variable(vv, name=kk)
        tparams[kk].trainable = t_trainable
    return tparams

def init_tparams(params):
    return update_tparams(tparams = None, params=params)

# load parameters
def load_params(path, params):
    pp = np.load(path)
    for kk, vv in pp.iteritems():
        #if kk not in params:
        #    raise Warning('%s is not in the archive' % kk)
        if kk not in params:
            params[kk] = npwrapper(pp[kk], trainable=True)
        else:
            if hasattr(params[kk], 'trainable'):
                params[kk] = npwrapper(pp[kk], trainable=params[kk].trainable)
            else:
                params[kk] = npwrapper(pp[kk], trainable=True)
    return params

def updateModuleInfo(options=None, tparams=None, prefix=None, module_identifier=None):
    ''' 
    update the current Module weights based on the tparams.
    '''
    thisModule = options[module_identifier]
    thisModule.build = True
    sub_Params = get_subdict_prefix(tparams, prefix)
    for _, v in sub_Params.iteritems():
        if getattr(v, 'trainable', True) is True:
            thisModule.trainable_weights.append(v)
        else:
            thisModule.non_trainable_weights.append(v)
    

def update_or_init_params(tparams=None, params=None, tmp_params=None):
    
    # we need to make sure tmp_params.trainable is passed through.
    tmp_params_keys = tmp_params.keys()
    if params is not None and set(tmp_params_keys).issubset(params.keys()):
        # if params is given, we update tparams from params
        for k,v in tmp_params.iteritems():
            trainable = v.trainable
            tmp_params[k] = params[k]
            tmp_params[k].trainable = trainable
        
        tparams = update_tparams(tparams, tmp_params) 
    else:
        # if params does not has all the keys, we need to initliaze it    
        if set(tmp_params_keys).issubset(tparams.keys()):
            # we don't need to do anything, just use tparams
            pass
        else:
            # if tparams does not has all the keys, it means we need to initlize params and then tparams
            tparams = update_tparams(tparams, tmp_params)
            if params is not None:
                for k,v in tmp_params.iteritems():
                    params[k] = v
    return tparams, params

def get_layer_identifier(prefix):
    return 'layer_' + prefix

def get_module_identifier(prefix):
    return 'module_' + prefix
        
class obj(object):
    pass

def init_LayerInfo(options, name):
    if not name  in options:
        thisModule = obj()
        thisModule.build = False
        thisModule.trainable_weights = []
        thisModule.non_trainable_weights = []
        thisModule.regularizers = []
        thisModule.constraints = []
        thisModule.fathers = []
        options[name] = thisModule
    return options

def init_ModuleInfo(options, name):
    if not name  in options:
        thisModule = obj()
        thisModule.build = False
        thisModule.trainable_weights = []
        thisModule.non_trainable_weights = []
        thisModule.regularizers = []
        thisModule.constraints = []
        thisModule.containers = []
        thisModule.fathers = []
        options[name] = thisModule
    return options

def build_or_not(module_identifier, options):
    '''To decide do you wanna build this module'''
    
    if not module_identifier in options:
        return True
    else:
        thisModule = options[module_identifier]
        return not thisModule.build
        
def update_father_module(options,belonging_Module, module_identifier):
    '''
    we need to update the father module which contains this layer based on the information
    in this layer
    '''
    if belonging_Module is not None:
        # means we need to update the father module info mation here
        if not belonging_Module in options:
            warnings.warn('father module: {m} not initialized before calling: {f}'.
                         format(m=belonging_Module,f=module_identifier))
            init_ModuleInfo(options, belonging_Module)
        thisInfo = options[module_identifier]
        thisInfo.fathers.append(belonging_Module)
        
        belongingModuleInfo = options[belonging_Module]
        belongingModuleInfo.trainable_weights += thisInfo.trainable_weights
        belongingModuleInfo.non_trainable_weights += thisInfo.non_trainable_weights
        belongingModuleInfo.regularizers += thisInfo.regularizers
        belongingModuleInfo.constraints  += thisInfo.constraints
        belongingModuleInfo.containers.append(module_identifier)
        return options 

def get_subdict_prefix(tparams, prefixlist=None):
    ''' 
    this function is used to substract the param dictioary based on the list of prefixlist.
    '''
    if not isinstance(prefixlist, list):
        prefixlist = [prefixlist]
    prefixlist = tuple(prefixlist)
    subParams = OrderedDict()

    for k in tparams.keys():
        if k.startswith(prefixlist):
            subParams[k] = tparams[k]
    return subParams

def load_keras_model(params, keras_model_path, max_layer = None):
    '''
    Load all layer weights from a HDF5 save file.
    '''
    if params is None:
        params = OrderedDict()
    import h5py
    f = h5py.File(keras_model_path, mode='r')    
    # new file format
    layer_names = [n.decode('utf8') for n in f.attrs['layer_names']] 
    if max_layer is None:
        max_layer = len(layer_names)   
    for k, name in enumerate(layer_names):
        if k == max_layer:
            break
        g = f[name]
        weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
        if len(weight_names):
            for weight_name in weight_names:
                if weight_name not in params:
                    warnings.warn("params does not have key: {s} when loading keras \
                                   model".format(s=weight_name))
                params[weight_name] = npwrapper(g[weight_name], trainable=True) 
    f.close()    
    return params
            
    
def weighted_objective(fn):
    '''Transforms an objective function `fn(y_true, y_pred)`
    into a sample-weighted, cost-masked objective function
    `fn(y_true, y_pred, weights, mask)`.
    '''
    def weighted(y_true, y_pred, weights, mask=None):
        # score_array has ndim >= 2
        score_array = fn(y_true, y_pred)
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            mask = T.cast(mask, K.floatx())
            # mask should have the same shape as score_array
            score_array *= mask
            #  the loss per batch should be proportional
            #  to the number of unmasked samples.
            score_array /= T.mean(mask)

        # reduce score_array to same ndim as weight array
        ndim = T.ndim(score_array)
        weight_ndim = T.ndim(weights)
        score_array = T.mean(score_array, axis=list(range(weight_ndim, ndim)))

        # apply sample weighting
        if weights is not None:
            score_array *= weights
            score_array /= T.mean(K.cast(K.not_equal(weights, 0), K.floatx()))
        return T.mean(score_array)
    return weighted    

import re
def split_words(words):
    return re.findall(r'\w+|\S+', words)

# The following functions are for 
def slice_tensor(x, n, dim):
    if T.ndim(x) == 3:
        return x[:, :, n*dim:(n+1)*dim]
    elif T.ndim(x) == 2:
            return x[:, n*dim:(n+1)*dim]
    return x[n*dim:(n+1)*dim]

def read( w, M):
    return (w[:, :, None]*M).sum(axis=1)

def get_content_w( beta, k, M):
    num = beta[:, None] * cosine_distance(M, k)
    return T.softmax(num)

def get_location_w(g, s, C, gamma, wc, w_tm1):
    wg = g[:, None] * wc + (1-g[:, None])*w_tm1
    Cs = (C[None, :, :, :] * wg[:, None, None, :]).sum(axis=3)
    wtilda = (Cs * s[:, :, None]).sum(axis=1)
    wout = renorm(wtilda ** gamma[:, None])
    return wout

def get_controller_output(h, W_k, b_k, W_c, b_c, W_s, b_s, k_activ = T.tanh):
    k = k_activ(T.dot(h, W_k) + b_k)  # + 1e-6
    #k = theano.printing.Print('[Debug] k shape is: ', attrs=("shape",))(k)
    c = T.dot(h, W_c) + b_c
    beta = T.softplus(c[:, 0]) + 1e-4
    g = T.sigmoid(c[:, 1])
    gamma = T.softplus(c[:, 2]) + 1.0001
    s = T.softmax(T.dot(h, W_s) + b_s)
    return k, beta, g, gamma, s
