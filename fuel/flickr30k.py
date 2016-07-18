import pickle as pkl
import gzip
import os
import sys
import time
from collections import OrderedDict
import numpy

def prepare_data(caps, features, worddict, maxlen=None, n_words=10000, zero_pad=False):
    # x: a list of sentences
    seqs = []
    feat_list = []
    for cc in caps:
        seqs.append([worddict[w] if worddict[w] < n_words else 1 for w in cc[0].split()])
        feat_list.append(features[cc[1]])

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
        y[idx,:] = numpy.array(ff.todense())
    y = y.reshape([y.shape[0], 14*14, 512])
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




    
def load_data(load_train=True, load_dev=True, load_test=True, path= '../Data/TrainingData/flickr30k'):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset
    '''

    #############
    # LOAD DATA #
    #############
    print '... loading data'
    from collections import OrderedDict
    
    train_cap = OrderedDict()
    if load_train:
        with open(os.path.join(path,'flicker_30k_align.train.pkl'), 'rb') as f:
            train_cap = pkl.load(f)
            train_feat = pkl.load(f)
        train = (train_cap, train_feat)
    else:
        train = None
    test_cap = OrderedDict()
    if load_test:
        with open(os.path.join(path,'flicker_30k_align.test.pkl'), 'rb') as f:
            test_cap = pkl.load(f)
            test_feat = pkl.load(f)
        test = (test_cap, test_feat)
    else:
        test = None
        
    dev_cap = OrderedDict()
    if load_dev:
        with open(os.path.join(path,'flicker_30k_align.dev.pkl'), 'rb') as f:
            dev_cap = pkl.load(f)
            dev_feat = pkl.load(f)
        valid = (dev_cap, dev_feat)
    else:
        valid = None
    
    #with open(os.path.join(path,'dictionary_old.pkl'), 'rb') as f:
    #    worddict = pkl.load(f)
    
    
    
    worddict = OrderedDict()  
    
    worddict['<eos>'] = 0
    worddict['UNK'] = 1
    wordIndex = 2
    
    total_cap= train_cap + test_cap + dev_cap
    for this_keys in total_cap:
        words = this_keys[0].split()             
        for k in words:
            if k not in worddict:
               worddict[k] = wordIndex
               wordIndex = wordIndex + 1
        
    
    with open(os.path.join(path,'dictionary.pkl'), 'wb') as f:
        pkl.dump(worddict, f, protocol=pkl.HIGHEST_PROTOCOL)
    return train, valid, test, worddict
