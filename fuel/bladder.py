import cPickle as pkl
import os
from collections import OrderedDict
from preprocessing.sequence import pad_sequences
import numpy as np
import re
from Core.utils_func import split_words
from utils.local_utils import *
import json
from backend.export import npwrapper
from fuel.Extractor import get_cnn_img

def prepare_data(caps, features, worddict, maxlen=None, n_words=1000, zero_pad=False, online_feature=False):
    # x: a list of sentences
    # caps is ('this is a good example.',0) the second term denotes the feature index
    seqs = []
    feat_list = []
    for cc in caps:
        seqs.append([worddict[w] if worddict[w] < n_words else 1 for w in split_words(cc[0])])
        feat_list.append(features[cc[1]: cc[1] + 1])

    lengths = [len(s) for s in seqs]
    if maxlen != None:
        new_seqs = []
        new_feat_list = []
        new_lengths = []
        for l, s, y in zip(lengths, seqs, feat_list):
            if l < maxlen:
                new_seqs.append(s)
                new_feat_list.append(y)
                new_lengths.append(l)
        lengths = new_lengths
        feat_list = new_feat_list
        seqs = new_seqs
        if len(lengths) < 1:
            return None, None, None
    
    if online_feature == False:
        feat = np.zeros((len(feat_list), feat_list[0].shape[1])).astype('float32')
        for idx, ff in enumerate(feat_list):
            #y[idx,:] = np.array(ff)
            feat[idx,:] = np.array(ff.todense())
        feat = feat.reshape([feat.shape[0], 512,14*14])
        feat = np.transpose(feat, (0, 2, 1))

        if zero_pad:
            feat_pad = np.zeros((feat.shape[0], feat.shape[1]+1, feat.shape[2])).astype('float32')
            feat_pad[:,:-1,:] = feat
            feat = feat_pad
    else:
        dest_shape = (224,224,3)
        shape = (3,224,224)
        feat = np.zeros((len(feat_list),) + shape).astype('float32')
       
        for idx, ff in enumerate(feat_list):
            #y[idx,:] = np.array(ff)
            ff = ff[0]
            feat[idx,:] = get_cnn_img(ff,dest_shape,local_norm=True)
        #print feat.shape
    n_samples = len(seqs)
    maxlen = np.max(lengths)+1
    #because we need to make it has one end of sentence in the end, so one more symbol.
    caps_x = np.zeros((maxlen, n_samples)).astype('int64')
    caps_mask = np.zeros((maxlen, n_samples)).astype('float32')
    for idx, s in enumerate(seqs):
        caps_x[:lengths[idx],idx] = s
        caps_mask[:lengths[idx]+1,idx] = 1.

    #caps, caps_mask = pad_sequences(seqs, maxlen=None, dtype='int64',
    #             padding='post', truncating='pre', value=0.)
    #caps = np.transpose(caps, (1,0))
    #caps_mask = np.transpose(caps_mask, (1,0))

    return npwrapper(caps_x), npwrapper(caps_mask), npwrapper(feat)

from Extractor import ListExtractor as dataExtractor
import cPickle

def get_anno(classparams, filepath = None, chunknum = 5000, batchsize=32, get_img=True):
    StruExtractor = dataExtractor(classparams)
    datainfo = StruExtractor.datainfo
    Totalnum = datainfo['Totalnum']
    
    totalIndx = np.arange(Totalnum)
    numberofchunk = (Totalnum + chunknum - 1)// chunknum   # the floor
    chunkstart = 0 
    
    thisanno = [None for _ in range(chunknum)]
    Total_anno = []
    for chunkidx in range(numberofchunk):
        thisnum = min(chunknum, Totalnum - chunkidx*chunknum)
        thisInd = totalIndx[chunkstart: chunkstart + thisnum]
        chunkstart += thisnum

        StruExtractor.getImg_Anno(thisInd, thisbatch[0:thisnum], thisanno, get_img=False)
        Total_anno.extend(thisanno[0:thisnum])
    if filepath is not None:
        with open(filepath, 'wb') as f:
            cPickle.dump(Total_anno, f,protocol=cPickle.HIGHEST_PROTOCOL)
    return Total_anno
    
def group_cap(cap):
    returned_cap = []
    orderedCap = OrderedDict()
    for thiscap in cap:
        ind = thiscap[1]
        thiswords = thiscap[0]
        if ind not in orderedCap:
            orderedCap[ind] = thiswords
        else:
            orderedCap[ind] = orderedCap[ind] + ' ' + thiswords
    for key in  orderedCap.keys():
        returned_cap.append( (orderedCap[key], key)  )
    return returned_cap

def load_data(load_train=True, load_dev= True, load_test= False, root_path='../Data/TrainingData/bladder', 
              img_ext='.png', online_feature=False):
    ''' Loads the dataset
    :type dataset: string
    :param dataset: the path to the dataset
    '''
    print '... loading data. ...'
    from scipy import sparse
    
    cap_root_path = os.path.join(root_path, 'Feat_conv')
    img_root_path = os.path.join(root_path, 'Img')

    feat_len = 512*14*14
    train_cap = []

    train_cap_pkl = os.path.join(cap_root_path , 'bladder_align_train_cap.pkl')

    classparams = {}
    classparams['datadir']   =  trainingimagefolder
    classparams['annodir']  =   annodir
    classparams['dataExt']   =  ['.png']             # the data ext
    classparams['nameList']  =  train_file_list
    classparams['labelList'] =  train_label_list 
    classparams['destin_shape']   =  (224,224) 
    classparams['channel']   =  img_channels


    training_classparams = classparams.copy()
    if not os.exists(train_cap_pkl):
        with open(trainingSplitFile) as data_file:    
            trainingSplitDict = json.load(data_file)
            train_file_list =  trainingSplitDict['img']
        training_classparams['nameList']  =  train_file_list
        get_anno(training_classparams, saveFolder=cap_root_path, filename='bladder_align_train_cap.pkl',
                 chunknum = 5000, batchsize=32, get_img=False)
    
    testing_classparams = classparams.copy()
    if not os.exists(train_cap_pkl):
        with open(trainingSplitFile) as data_file:    
            testingSplitDict = json.load(data_file)
            test_file_list =  testingSplitDict['img']
        testing_classparams['nameList']  =  test_file_list
        get_anno(training_classparams, saveFolder=cap_root_path, filename='bladder_align_test_cap.pkl',
                 chunknum = 5000, batchsize=32, get_img=False)


    if load_train:    
        with open(train_cap_pkl, 'rb') as ft_cap:
            train_cap = pkl.load(ft_cap)
            returned_train_cap = train_cap #group_cap(train_cap)

        if online_feature == False:
            feat_root_path = cap_root_path
            train_feat = None
            #train_feat = np.ones((len(returned_train_cap), feat_len))
            train_feat = sparse.csr_matrix((0, feat_len))
            current = 0
            for idx in range(9):
                print(idx)
                filename = 'bladder_align_train' + str(idx) + '.pkl'           
                with open(os.path.join(feat_root_path, filename), 'rb') as ft_feat:
                    #train_feature_list.append(pkl.load(ft_feat).todense())
                    thistemp = pkl.load(ft_feat)
                    train_feat =  sparse.vstack((train_feat, thistemp))
                    current = current + thistemp.shape[0]       
            assert current == len(returned_train_cap)
        else:
            train_feat = []
            for cap_tuple in returned_train_cap:
                name = cap_tuple[2]
                train_feat.append(os.path.join(img_root_path, name+img_ext))
        
        train = (returned_train_cap, train_feat)
    else:
        train = None
    
    #------Load testing set-----------------

    if load_test:
        test_cap = []
        with open(os.path.join(cap_root_path , 'bladder_align_test_cap.pkl'), 'rb') as ft_cap:
            test_cap = pkl.load(ft_cap)
        returned_test_cap = test_cap #group_cap(test_cap)

        if online_feature == False:
            feat_root_path = cap_root_path
            with open(os.path.join(feat_root_path,'bladder_align_test0.pkl'), 'rb') as f:
                test_feat = pkl.load(f)
            returned_test_cap = group_cap(test_cap)
            
        else:     
            test_feat = []
            for cap_tuple in returned_test_cap:
                name = cap_tuple[2]
                test_feat.append(os.path.join(img_root_path, name+img_ext))

        test = (returned_test_cap, test_feat)

    else:
        test = None

    valid_cap = []    
    if load_dev:
        with open(os.path.join(cap_root_path , 'bladder_align_test_cap.pkl'), 'rb') as ft_cap:
            valid_cap = pkl.load(ft_cap)
        returned_valid_cap = valid_cap #group_cap(valid_cap)
        
        if online_feature == False:
            feat_root_path = cap_root_path
            valid_feat = sparse.csr_matrix((0, feat_len))
            for idx in range(1):
                filename = 'bladder_align_test' + str(idx) + '.pkl'
                with open(os.path.join(feat_root_path, filename), 'rb') as ft_feat:
                    #valid_feature_list.append(pkl.load(ft_feat).todense())
                    thistemp = pkl.load(ft_feat)
                    valid_feat =  sparse.vstack((valid_feat, thistemp))
        else:
            valid_feat = []
            for cap_tuple in returned_valid_cap:
                name = cap_tuple[2]
                valid_feat.append(os.path.join(img_root_path, name+img_ext))


        valid = (returned_valid_cap, valid_feat)
    else:
        valid = None
    
    #with open(os.path.join(cap_root_path,'dictionary.pkl'), 'rb') as f:
    #    worddict = pkl.load(f)

    worddict = OrderedDict()
    worddict['<eos>'] = 0
    worddict['UNK'] = 1
    wordIndex = 2

    total_cap= train_cap + test_cap + valid_cap
    for this_keys in total_cap:
        words = split_words(this_keys[0])
        for k in words:
            if k not in worddict:
                worddict[k] = wordIndex
                wordIndex = wordIndex + 1

    with open(os.path.join(path,'dictionary.pkl'), 'wb') as f:
        pkl.dump(worddict, f, protocol=pkl.HIGHEST_PROTOCOL)

    return train, valid, test, worddict
