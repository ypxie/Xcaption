
import numpy as np
import copy
from Core.utils_func import split_words
import random

class HomogeneousData():
    def __init__(self, data, batch_size=128, maxlen=None):
        self.batch_size = 128
        self.data = data
        self.batch_size = batch_size
        self.maxlen = maxlen

        self.prepare()
        self.reset()

    def prepare(self):
        self.caps = self.data[0]

        self.chunkstart = 0
        self.Totalnum = len(self.caps)
        sample_rand_Ind = range(self.Totalnum)
        random.shuffle(sample_rand_Ind)
        self.totalIndx = sample_rand_Ind

        self.numberofchunk = (self.Totalnum + self.batch_size - 1) // self.batch_size   # the floor
        self.chunkidx = 0

    def reset(self):
        self.prepare()

    def next(self):
        
        thisnum = min(self.batch_size, self.Totalnum - self.chunkidx*self.batch_size)
        curr_indices = self.totalIndx[self.chunkstart: self.chunkstart + thisnum]

        self.chunkstart += thisnum
        self.chunkidx += 1

        caps = [self.caps[ii] for ii in curr_indices]
        if self.chunkidx > self.numberofchunk:
            self.reset()
            raise StopIteration()
        return caps
            

    def __iter__(self):
        return self
