# -*- coding: utf-8 -*-
from Core.recurrent import *
from Core.simple import *
import numpy as np
"""
Neural network layer definitions.

The life-cycle of each of these layers is as follows
    1) The param_init of the layer is called, which creates
    the weights of the network.
    2) The fprop is called which builds that part of the Theano graph
    using the weights created in step 1). This automatically links
    these variables to the graph.

Each prefix is used like a key and should be unique
to avoid naming conflicts when building the graph.
"""

layers = {'ff': ('param_init_fflayer', 'fflayer'),
          'lstm': ('param_init_lstm', 'lstm_layer'),
          'lstm_cond': ('param_init_dynamic_lstm_cond', 'dynamic_lstm_cond_layer'), #'lstm_cond_layer'),
          }

def get_layer(name):
    fns = layers[name]
    return (eval(fns[0]), eval(fns[1]))


