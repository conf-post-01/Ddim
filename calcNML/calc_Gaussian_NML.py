# -*- coding: utf-8 -*-

import sys
import os
import numpy as np
import pandas as pd
import math
import time
import json
from datetime import datetime as dt

class calcGMMNML:
    '''Calculate Gaussian NML log likelihood
    '''

    def __init__(self):
        '''constructor
        '''
        self.BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.DATA_DIR = os.path.normpath(os.path.join(self.BASE_DIR, './data'))
        self.OUT_DIR = os.path.normpath(os.path.join(self.DATA_DIR, './out'))

    def calc_log_likelihood(self,X,Z):
        '''Calculate log likelihood
        [Parameters]
            X: array like, shape ( # of data, dimension)
                Input dataset
            Z: list, shape ( # of data)
                The cluster indices of data points
        '''
        cluster_list = set(Z)
        data = pd.DataFrame(X)
        cluster = np.array(Z)
        N = len(cluster)
        dim = data.shape[1]
        log_likelihood = 0
        none_flg = False
        for c in cluster_list:
            ### Calculate parameters in each cluster
            d_list = (cluster==c)
            c_data = np.array(data.ix[d_list,:])
            c_N = sum(d_list)
            if c_N <= data.shape[1]:
                none_flg = True
                continue
            c_mu = np.mean(c_data) # vector of mean
            c_cov = np.cov(c_data.T) # variance-covariance matrix
            c_cov_inv = np.linalg.inv(c_cov) # inverse of c_cov

            ### calculate log likelihood of each cluster
            log_likelihood += c_N*np.log(float(c_N)/float(N))
            log_likelihood += -( c_N*dim/2.*np.log(2*math.pi*math.e) + c_N/2.*np.log(np.sqrt(np.linalg.det(c_cov))) )

        if none_flg:
            log_likelihood = np.nan

        return log_likelihood
