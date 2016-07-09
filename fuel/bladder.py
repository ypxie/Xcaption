import pickle as pkl
import gzip
import os
import sys
import time
from collections import OrderedDict
import numpy

def prepare_data(caps, features, worddict, maxlen=None, n_words=1000, zero_pad=False):
    # x: a list of sentences
    seqs = []
    feat_list = []
    for cc in caps:
        seqs.append([worddict[w] if worddict[w] < n_words else 1 for w in cc[0].split()])
        feat_list.append(features[cc[1]: cc[1] + 1,:])

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

    y = numpy.zeros((len(feat_list), feat_list[0].shape[1])).astype('float32')
    for idx, ff in enumerate(feat_list):
        y[idx,:] = numpy.array(ff)        
        #y[idx,:] = numpy.array(ff.todense())
    y = y.reshape([y.shape[0], 512,14*14])
    y = numpy.transpose(y,(0,2,1))

    if zero_pad:
        y_pad = numpy.zeros((y.shape[0], y.shape[1]+1, y.shape[2])).astype('float32')
        y_pad[:,:-1,:] = y
        y = y_pad

    n_samples = len(seqs)
    maxlen = numpy.max(lengths)+1

    x = numpy.zeros((maxlen, n_samples)).astype('int64')
    x_mask = numpy.zeros((maxlen, n_samples)).astype('float32')
    for idx, s in enumerate(seqs):
        x[:lengths[idx],idx] = s
        x_mask[:lengths[idx]+1,idx] = 1.

    return x, x_mask, y

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

def load_data(load_train=True, load_dev=False, load_test=False, path='../Data/TrainingData/bladder'):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset
    '''
    #############
    # LOAD DATA #
    #############
    print '... loading data'
    import deepdish as dd
    import scipy
    train_cap = []
    if load_train:
        with open(os.path.join(path , 'bladder_align.train.pkl'), 'rb') as ft_cap:
            train_cap = pkl.load(ft_cap)
        returned_train_cap = group_cap(train_cap)

        train_feat = None
       # train_feature_list = []
        train_feat = numpy.ones((len(returned_train_cap), 512*14*14))
        
        current = 0
        for idx in range(11):
            print(idx)
            filename = 'bladder_align.train' + str(idx) + '.pkl'           
            with open(path+ filename, 'rb') as ft_feat:
                #train_feature_list.append(pkl.load(ft_feat).todense())
                thistemp = pkl.load(ft_feat).todense()
                train_feat[current:current+thistemp.shape[0], :] = thistemp
                current = current + thistemp.shape[0]
                
        assert current == len(returned_train_cap)
        #train_feat = numpy.concatenate(train_feature_list, axis =0)            
        #train_feat = scipy.sparse.csr_matrix(train_feat)
        train = (returned_train_cap, train_feat)
    else:
        train = None
    test_cap = []
    if load_test:
        with open(os.path.join(path,'test.pkl'), 'rb') as f:
            test_cap = pkl.load(f)
            test_feat = pkl.load(f)

        returned_test_cap = group_cap(test_cap)
        test = (returned_test_cap, test_feat)
    else:
        test = None
    dev_cap = []    
    if load_dev:
        with open(os.path.join(path , 'bladder_align.val.pkl'), 'rb') as ft_cap:
            dev_cap = pkl.load(ft_cap)
        returned_dev_cap = group_cap(dev_cap)

        dev_feat = None
        #valid_feature_list = []
        for idx in range(2):
            filename = 'bladder_align.val' + str(idx) + '.pkl'
            with open(path+filename, 'rb') as ft_feat:
                #valid_feature_list.append(pkl.load(ft_feat).todense())
                if dev_feat is not None:
                   dev_feat = numpy.concatenate([dev_feat, pkl.load(ft_feat).todense()], axis = 0)
                else:
                   dev_feat = pkl.load(ft_feat).todense()
        #dev_feat = numpy.concatenate(valid_feature_list, axis =0)                    
        #dev_feat = scipy.sparse.csr_matrix(dev_feat)
        valid = (returned_dev_cap, dev_feat)
    else:
        valid = None

    with open(os.path.join(path,'dictionary.pkl'), 'rb') as f:
        worddict = pkl.load(f)

#    train_dict = {
#            'caption': returned_train_cap,
#            'feature': train_feat
#            }
#
#    valid_dict = {
#            'caption': returned_dev_cap,
#            'feature': dev_feat
#            }
#
#    dd.io.save(path+'train_file.h5',train_dict)
#    dd.io.save(path+'valid_file.h5',valid_dict)
#    
#    print('Finished writting data to the dataset')
#    worddict = OrderedDict()
#    worddict['<eos>'] = 0
#    worddict['UNK'] = 1
#    wordIndex = 2
#
#    total_cap= train_cap + test_cap + dev_cap
#    for this_keys in total_cap:
#        words = this_keys[0].split()
#        for k in words:
#            if k not in worddict:
#               worddict[k] = wordIndex
#               wordIndex = wordIndex + 1
#
#    with open(os.path.join(path,'dictionary.pkl'), 'wb') as f:
#        pkl.dump(worddict, f, protocol=pkl.HIGHEST_PROTOCOL)
#    
#    return train, valid, test, worddict
