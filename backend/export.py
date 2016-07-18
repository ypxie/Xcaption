# -*- coding: utf-8 -*-
#from numpy import *
#from theano.tensor import *
import os

if os.environ['debug_mode'] == 'True':
   from backend.numpy_backend import *
else:
   from backend.theano_backend import *

from backend.numpy_backend import npwrapper