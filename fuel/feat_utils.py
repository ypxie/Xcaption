# -*- coding: utf-8 -*-
import sys
#from Core.simple import *
from collections import OrderedDict

import backend.export as T

from Core.engine import  get_conv_feature
from Core.utils_func import load_keras_model
keras_model_path = '/home/yuanpuxie/DataSet/Bladder_Caption/Augmented/Feat_conv/weights.h5'

def make_feat_func(keras_model_path):
    options = dict()
    params = OrderedDict()
    tparams = OrderedDict()
    # load weights here
    
    load_keras_model(params, keras_model_path, max_layer = 12)

    inputs = T.placeholder(shape=(None,3,None,None))
    feat_pool = get_conv_feature(tparams, options, inputs, params = params, trainable=False)

    conv_5 = feat_pool[0]
    feat_extractor = T.function([inputs, T.learning_phase()],[conv_5],allow_input_downcast=True)
    print('Finished making feature extracting function!')
    return feat_extractor

from Extractor import ListExtractor as dataExtractor
from scipy import sparse
import cPickle

def get_feat(data,batchsize):
 
    Totalnum = data.shape[0]
    totalIndx = np.arange(Totalnum)
    
    numberofchunk = (Totalnum + batchsize - 1)// batchsize   # the floor
    chunkstart = 0 
    #print('numberofchunk: {s}'.format(s=str(numberofchunk)))
    sample_size = feat_extractor(data[0:1],0)[0].shape[1:]
    print (Totalnum,)+ sample_size
    Total_data = np.zeros((Totalnum,)+ sample_size )
    
    for chunkidx in range(numberofchunk):
        thisnum = min(batchsize, Totalnum - chunkidx*batchsize)
        thisInd = totalIndx[chunkstart: chunkstart + thisnum]  
        this_slice = slice(chunkstart, chunkstart + thisnum,1)
        
        chunkstart += thisnum    
        batch_data = data[this_slice]
        #print batch_data.shape
        output = feat_extractor(batch_data,0)[0]
        Total_data[this_slice,...] = output
        
    return Total_data

def saveFeat(classparams, saveFolder='.', basename='bladder_align.train',cap_ext='_cap',chunknum = 5000, batchsize=32):
    StruExtractor = dataExtractor(classparams)
    datainfo = StruExtractor.datainfo
    Totalnum = datainfo['Totalnum']
    
    totalIndx = np.arange(Totalnum)
    numberofchunk = (Totalnum + chunknum - 1)// chunknum   # the floor
    chunkstart = 0 
    
    thisbatch = np.zeros((chunknum,) + tuple(datainfo['inputshape'] ))
    thisanno = [None for _ in range(chunknum)]
    Total_anno = []
    for chunkidx in range(numberofchunk):
        thisnum = min(chunknum, Totalnum - chunkidx*chunknum)
        thisInd = totalIndx[chunkstart: chunkstart + thisnum]
        chunkstart += thisnum

        StruExtractor.getImg_Anno(thisInd, thisbatch[0:thisnum], thisanno)
        Total_anno.extend(thisanno[0:thisnum])
        
        featStacks =  get_feat(thisbatch[0:thisnum],batchsize)
        featStacks = featStacks.reshape((featStacks.shape[0], -1))
        
        filename = basename + str(chunkidx) + '.pkl'    
        filepath = os.path.join(saveFolder,filename)
        with open(filepath,'wb') as f:
            cPickle.dump(sparse.csr_matrix(featStacks), f,protocol=cPickle.HIGHEST_PROTOCOL)
            print("Success!")
    
    filename = basename + cap_ext + '.pkl'    
    filepath = os.path.join(saveFolder,filename)
    print featStacks.shape
    with open(filepath, 'wb') as f:
        cPickle.dump(Total_anno, f,protocol=cPickle.HIGHEST_PROTOCOL)
    print('Done')


img_channels = 3
chunknum = 50000

classparams = {}
classparams['datadir']   =  trainingimagefolder
classparams['annodir']  =   annodir
classparams['dataExt']   =  ['.png']             # the data ext
classparams['nameList']  =  train_file_list
classparams['labelList'] =  train_label_list 
classparams['destin_shape']   =  (224,224) 
classparams['channel']   =  img_channels

saveFolder = os.path.join(dataroot,'Feat_conv')
print('get training data done')
saveFeat(classparams, saveFolder, basename = 'bladder_align_train',chunknum = chunknum)    
print('Done')
