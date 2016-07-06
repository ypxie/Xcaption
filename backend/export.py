# -*- coding: utf-8 -*-
#from numpy import *
#from theano.tensor import *

#from backend.numpy_backend import *

from backend.theano_backend import *

class npwrapper(np.ndarray):
    '''usage: to append trainable attr to numpy object in layer initialization
       eg: b = npwrapper(np.arange(5), trainable=False) '''
    def __new__(cls, input_array, trainable=True):
        obj = np.asarray(input_array).view(cls)
        obj.trainable = trainable
        return obj 

    def __array_finalize__(self, obj):
        #print('In __array_finalize__:')
        #print('   self is %s' % repr(self))
        #print('   obj is %s' % repr(obj))
        if obj is None: return
        self.trainable = getattr(obj, 'trainable', None)

    def __array_wrap__(self, out_arr, context=None):
        #print('In __array_wrap__:')
        #print('   self is %s' % repr(self))
        #print('   arr is %s' % repr(out_arr))
        # then just call the parent
        return np.ndarray.__array_wrap__(self, out_arr, context)

