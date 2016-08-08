# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, '/home/yuanpuxie/OwnCloud/WorkStation/Medical_Caption/Code')
from Core.simple import *
from collections import OrderedDict

import backend.export as T

from Core.engine import  get_conv_feature
from Core.utils_func import load_keras_model

options = dict()
params = OrderedDict()
tparams = OrderedDict()
# load weights here
keras_model_path = '/home/yuanpuxie/OwnCloud/WorkStation/fullyConv/Data/Model/Bladder/conclusion_bladder/weights.h5'
load_keras_model(params, keras_model_path, max_layer = 12)

inputs = T.placeholder(shape=(None,3,None,None))
feat_pool = get_conv_feature(tparams, options, inputs, params = params, trainable=False)

conv_5 = feat_pool[0]
feat_extractor = T.function([inputs, T.learning_phase()],[conv_5])



