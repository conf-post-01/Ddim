# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
import pandas as pd
from scipy.linalg import cholesky
import scipy.stats as sp

import os
import sys
import random
import copy
import json
import time
import itertools
from sklearn.preprocessing import normalize
from datetime import datetime as dt

class gmmGenerator():
    '''Generator for time-series of Gaussian Mixture Model
    [Parameters]
        dim: int, optional (default:3)
            The dimension of Gaussian Mixture Model

        num: int, optional (default:100)
            The number of dataset in each time

        l_T: list, optional (default:[10,10,10])
            The length of [ before-change, transition-period, after-change]

        l_K: list, optional (default:[2,3,3])
            The number of clusters in each time [ before-change, transition-period, after-change]

        l_mu: list, optional (default:None)
            The center in each cluster
            (ex. [[10,10,10],[-10,-10,-10],[0,0,0]])

        var: float, optional (default:2)
            The scale of variance-covariance matrix

        corr: float, optional (default:0.2)
            The scale of correlation

        abs_mu: float, optional (default:10)
            The scale of center in each cluster

        random_state: int, optional (default:0)
            The state of random element

        change_alpha: float, optional (default:1)
            Define the change type (if 1, linear change)
    '''

    def __init__(self, dim=3, num=100, l_T=[10,10,10], l_K=[2,3,3], l_mu=None,
                 var=2, corr=.2, abs_mu=10, random_state=0, change_alpha=1.):
        '''constructor
        '''

        self.BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.DATA_DIR = os.path.normpath(os.path.join(self.BASE_DIR, './data'))
        self.OUT_DIR = os.path.normpath(os.path.join(self.DATA_DIR, './out'))

        ### parameter for sample dataset
        self.dim = dim
        self.num = num
        self.l_T = np.cumsum(l_T).tolist()
        self.l_K = l_K
        self.true_k = []
        for i,(t,k) in enumerate(zip(l_T,l_K)):
            if i == 1:
                self.true_k.extend([np.nan]*t)
                continue
            self.true_k.extend([k]*t)
        self.l_mu = l_mu
        self.var = var
        self.corr = corr # 0 <= corr <= 1, the bigger corr means the weaker correlation
        self.abs_mu = abs_mu
        self.random_state = random_state
        self.rand_obj = np.random.RandomState(seed=self.random_state)
        self.change_alpha = change_alpha

        self.params = {'dim':self.dim,'N':self.num,'l_T':self.l_T,'l_K':self.l_K,
                        'l_mu':self.l_mu,'var':self.var,'corr':self.corr}

        ### generate center of dataset
        self._create_mean()

    def _create_mean(self):
        '''Create center in each cluster
        '''
        if self.l_mu == None:
            choice_list = list(itertools.product((1,-1), repeat=3))
            l_mu = np.array(choice_list)[self.rand_obj.choice(range(len(choice_list)),size=5)].tolist()
            l_mu = ( np.array(l_mu) * self.abs_mu ).tolist()
            self.l_mu = l_mu

    def _create_varcov(self):
        '''Create variance-covariance matrix in each cluster
        '''
        C_list = []
        for i in range(np.max(self.l_K)):
            A = normalize(self.rand_obj.normal(size=(self.dim,self.dim)),axis=1)
            C = self.var * ( self.corr * np.dot(A,A.T) + (1-self.corr) * np.eye(self.dim) )
            C_list.append(C)

        return C_list

    def _d_generate_gradual_gaussian(self,c_ind=None):
        '''Create time-series of dataset with Gaussian Mixture Model
        '''

        ### center in each cluster
        af_mu = np.array(self.l_mu[2])
        bef_mu = np.array(self.l_mu[1])

        C_list = self._create_varcov()

        l_x = []
        l_ind = []
        for t in range(self.l_T[-1]):
            ### define the number of clusters & parameter lambda
            if t < self.l_T[0]:
                K = self.l_K[0]
                mu = self.l_mu[:K]
                l_C = C_list[:K]
            elif t < self.l_T[2]:
                K = self.l_K[2]
                mu = self.l_mu
                l_C = C_list

            ### define the cluster index
            if t == 0:
                if c_ind is None:
                    c_ind = [np.int64(self.rand_obj.random_sample()*K) for _ in range(self.num)]
                c_bf_ind = copy.copy(c_ind)
            elif self.l_T[0] == t:
                c_ind = copy.copy(c_bf_ind)
                change_list = [i for i in range(self.num)]

                p_new = 1./K
                new_list = [c for i,c in enumerate(change_list) if self.rand_obj.random_sample()<p_new ]
                new_ind = [np.int64(self.rand_obj.random_sample()*K) for _ in new_list]
                c_ind_tmp = np.array(c_ind)
                c_ind_tmp[new_list] = new_ind
                c_ind = list(c_ind_tmp)
                c_bf_ind = copy.copy(c_ind)

                ### current center of new cluster
                tmp_mu = bef_mu + (t-self.l_T[0]+1)**self.change_alpha / ((self.l_T[1]-t)**self.change_alpha+(t-self.l_T[0]+1)**self.change_alpha ) * ( af_mu - bef_mu )
                mu[2] = tmp_mu.tolist()

            elif ( self.l_T[0] < t ) and ( t <= self.l_T[1] ):
                c_ind = copy.copy(c_bf_ind)
                c_ind_tmp = np.array(c_ind)
                ### current center of new cluster
                tmp_mu = bef_mu + (t-self.l_T[0]+1)**self.change_alpha / ((self.l_T[1]-t)**self.change_alpha+(t-self.l_T[0]+1)**self.change_alpha ) * ( af_mu - bef_mu )
                mu[2] = tmp_mu.tolist()

            else:
                c_ind = copy.copy(c_bf_ind)
                c_bf_ind = copy.copy(c_ind)

            ### generate dataset
            x = self.create_gmm_data(mu_list=mu,C_list=l_C,num=self.num,clu_ind=c_ind)

            l_x.append(x.values.tolist())
            l_ind.append(c_bf_ind)

        return np.array(l_x), np.array(l_ind)

    def create_gmm_data(self,mu_list,C_list,num,clu_ind):
        '''Generate dataset with Gaussian Mixture Model
        [Parameters]
            mu_list: list, shape (k,m)
                List of center in each cluster

            C_list: list, shape (k,m,m)
                List of variance-covariance matrix

            num: int
                The number of dataset

            clu_ind: list, shape (num)
                Cluster indices
        '''
        if num > 1:
            gmm_data = pd.DataFrame(np.zeros((num,self.dim)))
            k_num = len(set(clu_ind))
            clu_ind = np.array(clu_ind)
            for i in range(k_num):
                clu_num = sum(clu_ind==i)
                X = self.rand_obj.normal(size=(clu_num,self.dim))
                A = cholesky(C_list[i])
                gmm_data.loc[clu_ind==i] = np.dot(X,A) + np.array(mu_list[i])
        elif num == 1:
            gmm_data = pd.DataFrame(np.zeros((num,self.dim)))
            clu_ind = np.int64(np.floor(self.rand_obj.random_sample(num)*k_num))
            X = self.rand_obj.normal(size=(clu_num,self.dim))
            A = cholesky(C_list[0])
            gmm_data.loc[:,:] = np.dot(X,A) + np.array(mu_list[0])
        else:
            clu_ind = []
            gmm_data = pd.DataFrame(np.zeros((0,self.dim)))

        return gmm_data

    def generate(self,init_index=None):
        '''generate dataset
        '''

        data, ind = self._d_generate_gradual_gaussian(c_ind=init_index)

        self.data = data
        self.ind = ind

        return data, ind
