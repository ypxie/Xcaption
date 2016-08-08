from __future__ import absolute_import
import numpy as np
import os
from utils.local_utils import*

#from ImageGenerator import * 
from scipy.io import loadmat
from PIL import Image, ImageDraw
#from skimage.color import rgb2gray
import skimage, skimage.morphology
import math
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.cluster import MiniBatchKMeans
from  scipy.ndimage.interpolation import rotate
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import json

class ListExtractor(object):
    def __init__(self, initial_data):
        self.datadir = "" 
        self.dataExt =   [".jpg",'.tif']   
        self.nameList = []  # should only contains image name with extension
        self.labelList = [] # should be one-hot coding or simply a list of int
        self.local_norm = True
        self.destin_shape = None
        self.datainfo = None
        self.annodir = None
        self.annoExt = '.json'
        self.anno_count = 0
        for key in initial_data:
            setattr(self, key,initial_data[key])
        
        self.nb_sample = len(self.labelList)   
        if type(self.labelList[0]) is list:
            self.nb_class = len(self.labelList[0])
        else:
            self.nb_class = max(self.labelList)
        
        super(ListExtractor, self).__init__(initial_data)
        self.getMatinfo()

    def _standardize_label(self, label):
        if type(label) in [int, float]:
            label = to_one_hot(int(label), self.nb_class)
            
        elif type(label) in [list, tuple]:
            label = np.asarray(label)
        
        if type(label) is not np.ndarray:
            raise Exception('Wrong label input as : {s}'.format(s =  str(type(label))) )   
        return label
    
    def getOneDataBatch_stru(self,thisRandIndx=None, thisbatch=None, thislabel=None):
        if thisRandIndx is None:
            thisRandIndx = np.arange(0, len(self.nameList))
        if thisbatch is None:
            if self.datainfo is None:
                self.getMatinfo()
            thisbatch = np.zeros((self.datainfo['Totalnum'],) + self.datainfo['inputshape']) 
        if thislabel is None:
            thislabel = np.zeros((self.datainfo['Totalnum'],) + self.datainfo['outputshape']) 
        thisnum = 0
        for ind in thisRandIndx:
            thisname = self.nameList[ind]
            thislabel_list = self.labelList[ind]

            valid = False
            for imgExt in self.dataExt:
                thisfile = thisname + imgExt
                thispath = os.path.join(self.datadir, thisfile)
                if os.path.isfile(thispath):
                   img = imread(thispath)
                   valid = True
                   break
            if valid:
          
                img  = pre_process_img(img, yuv = False,norm=self.local_norm)
                if self.destin_shape is not None:
                    shape = tuple(self.destin_shape) + (img.shape[2],)
                    img =  imresize_shape(img,shape)
                # transpose the img to order (channel, row, col)
                img = np.transpose(img, (2,0,1))
                thisbatch[thisnum,...] = img
                thislabel[thisnum,...] = self._standardize_label(thislabel_list)
                thisnum += 1
            else:
                print('Image: {s} not find'.format(s = thisname))
        return thisbatch, thislabel

    
        
    def getImg_Anno(self,thisRandIndx=None, thisbatch=None, thisanno=None):
        if thisRandIndx is None:
            thisRandIndx = np.arange(0, len(self.nameList))
        if thisbatch is None:
            if self.datainfo is None:
                self.getMatinfo()
            thisbatch = np.zeros((self.datainfo['Totalnum'],) + self.datainfo['inputshape']) 
        if thisanno is None:
            thisanno = [None for _ in range(self.datainfo['Totalnum'])]
        thisnum = 0
        for ind in thisRandIndx:
            thisname = self.nameList[ind]
            valid = False
            for imgExt in self.dataExt:
                thisfile = thisname + imgExt
                thispath = os.path.join(self.datadir, thisfile)
                if os.path.isfile(thispath):
                   valid = True
                   break
            if valid: 
                         
                if self.destin_shape is not None:
                    shape = tuple(self.destin_shape) + (img.shape[2],)
                img = get_cnn_img(thispath,shape, self.local_norm)  

                thisbatch[thisnum,...] = img
                cap_tuple = (self._get_parse_anno(thisname), self.anno_count, thisname)
                self.anno_count += 1
                thisanno[thisnum]= cap_tuple
                thisnum += 1
            else:
                print('Image: {s} not find'.format(s = thisname))
        return thisbatch, thisanno
    def _get_parse_anno(self, thisname):
        thisfile = thisname + self.annoExt
        thispath = os.path.join(self.annodir, thisfile)
        with open(thispath) as data_file:    
            anno_dict = json.load(data_file)
            thisstr = ''
            for k, v in anno_dict.iteritems():
                thisstr = thisstr + ' '+ k.title() +': '
                thisstr  = thisstr + ' ' + v[1]
        return thisstr
    def getMatinfo(self):
        datainfo = {}
        ind = 0
        thisname = self.nameList[ind]
        #thislabel = self.labelList[ind]
        for imgExt in self.dataExt:
                thisfile = thisname + imgExt
                thispath = os.path.join(self.datadir, thisfile)
                if os.path.isfile(thispath):
                   img = imread(thispath)
                   img  = pre_process_img(img, yuv = False,norm=self.local_norm)
                   if self.destin_shape is not None:
                    shape = tuple(self.destin_shape) + (img.shape[2],)
                    img =  imresize_shape(img,shape)
                    
                   img = np.transpose(img, (2,0,1))
                   valid = True
                   break
         
        datainfo['outputdim'] = self.nb_class
        datainfo['inputdim'] =  np.prod(img.shape)
        datainfo['outputshape'] = (self.nb_class,)
        datainfo['inputshape'] =  img.shape

        datainfo['h'] = img.shape[1]
        datainfo['w'] = img.shape[2]
        datainfo['channel'] = img.shape[0]
        datainfo['Totalnum'] = len(self.nameList)
        self.datainfo = datainfo
        self.TrainingMatinfo['datainfo'] = datainfo
        return self.TrainingMatinfo    

def get_cnn_img(thispath, shape, local_norm):
    img = imread(thispath)
    img  = pre_process_img(img, yuv = False,norm= local_norm)
    img =  imresize_shape(img,shape)
    # transpose the img to order (channel, row, col)
    img = np.transpose(img, (2,0,1))
    return img