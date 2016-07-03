import backend.export as T
from Core.common import npwrapper

from Core.utils_func import *
import numpy as np
from utils import activations, initializations, regularizers, constraints


def param_init_convlayer(options, params, input_shape, nb_filter, nb_row, nb_col,prefix='conv',
	             init='glorot_uniform',  weights=None, dim_ordering= 'th',
                 W_regularizer=None, b_regularizer=None, activity_regularizer=None,
                 W_constraint=None, b_constraint=None,bias=True, dilated = 0, 
                 rate = 1,trainable=True, **kwargs):

    init = initializations.get(init, dim_ordering=dim_ordering)
    if dim_ordering == 'th':
        stack_size = input_shape[1]
        W_shape = (nb_filter, stack_size, nb_row, nb_col)
    elif dim_ordering == 'tf':
        stack_size = input_shape[3]
        W_shape = (nb_row, nb_col, stack_size, nb_filter)
    else:
        raise Exception('Invalid dim_ordering: ' + dim_ordering)
       
    W = init(W_shape, dim_ordering = dim_ordering, symbolic=False)
    b = np.zeros((nb_filter))
    params[get_name(prefix, 'W')] = npwrapper(W, trainable=trainable)
    params[get_name(prefix, 'b')] = npwrapper(b, trainable=trainable)
    
    return params

def conv2dlayer(tparams, state_below, options, input_shape, prefix='rconv', 	       
            border_mode='valid', subsample=(1, 1), dim_ordering= 'th',
	        activation='linear', W_regularizer=None, 
	        b_regularizer=None, activity_regularizer=None,
            W_constraint=None, b_constraint=None,
            bias=True, dilated = 0, rate = 1,**kwargs
            ):
    activation_func = activations.get(activation) 
    if W_regularizer:
       W_regularizer.set_param(tparams[get_name(prefix,'W')])
       options['regularizers'].append(W_regularizer)

    if b_regularizer:
       b_regularizer.set_param(tparams[get_name(prefix,'b')])
       options['regularizers'].append(b_regularizer)
    
    if dim_ordering == 'th':
        stack_size = input_shape[1]
        W_shape = (nb_filter, stack_size, nb_row, nb_col)
    elif dim_ordering == 'tf':
        stack_size = input_shape[3]
        W_shape = (nb_row, nb_col, stack_size, nb_filter)

    output = T.conv2d(x, tparams[get_name(prefix,'W')], strides=subsample,
                          dilated = dilated, rate= rate,
                          border_mode= border_mode,
                          dim_ordering= dim_ordering,
                          filter_shape= W_shape)
    if self.dim_ordering == 'th':
        output += T.reshape(tparams[get_name(prefix,'b')], (1, self.nb_filter, 1, 1))
    elif self.dim_ordering == 'tf':
        output += T.reshape(tparams[get_name(prefix,'b')], (1, 1, 1, self.nb_filter))
    else:
        raise Exception('Invalid dim_ordering: ' + self.dim_ordering)
    
    output = activation_func(output)
    return output


def maxpooing2d(inputs, pool_size=(2,2), strides=(2,2), border_mode='same', dim_ordering='th'):
        output = T.pool2d(inputs, pool_size, strides,
                          border_mode, dim_ordering, pool_mode='max')
        return output


def upsampling2d(self, x, size=(2,2),mask=None,dim_ordering = 'th'):
        return T.resize_images(x, size[0], size[1],
                               dim_ordering = dim_ordering)

def Resize2D(X,  destin_shape,dim_ordering='th',mask=None):
    input_shape =  list(T.shape(X))
    tmp_output_shape = list(destin_shape)
    if dim_ordering == 'th':
        destsize = tmp_output_shape[2:4]
    elif dim_ordering == 'tf':
        destsize = tmp_output_shape[1:3]

    if dim_ordering == 'th':
        destsize = list(destsize)
        row_residual = (destsize[0] - input_shape[2])
        col_residual = (destsize[1] - input_shape[3])

    elif dim_ordering == 'tf':
        row_residual = (destsize[0] - input_shape[1])
        col_residual = (destsize[1] - input_shape[2])
    padding = [row_residual//2,col_residual//2,  row_residual - row_residual//2,  col_residual - col_residual//2]
    cropping = [(-row_residual)//2, (-col_residual)//2,  -(row_residual + (-row_residual)//2),  -(col_residual + (-col_residual)//2)]
    #result = K.ifelse(K.gt(row_residual, 0), K.spatial_2d_padding_4specify(X, padding = padding), K.spatial_2d_cropping_4specify(X, cropping = cropping))
    result = T.spatial_2d_padding_4specify(X, padding = padding)
    #result = theano.printing.Print('Finish calculating resize')(result + 1)
    return result