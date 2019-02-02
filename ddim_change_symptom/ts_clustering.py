# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
import pandas as pd
from scipy.linalg import cholesky
from sklearn.mixture import GaussianMixture
import scipy.stats as sp

import os
import sys
import random
import copy
import json
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from datetime import datetime as dt

path = os.path.join(os.path.dirname(__file__), '../')
sys.path.append(path)
import calcNML.calc_Gaussian_NML as cNML
import structural_entropy.dynamic_model_selection as dms


class gmmSeqClustering():
    '''Clustering in each time
    [Parameters]
        norm_file: string
            The file name of NML normalization term
            (The user have to create this file before using this class.)

        max_clu: int, optional (default:5)
            Search range of the number of clusters

        num_em: int, optional (default:3)
            Number of times of EM algorithm

        std_flg: boolean, optional (default:True)
            Whether to standerize the dataset or not

        sdms_lam: float, optional (default:1)
            Scale of change code-length in SDMS

        sdms_beta: float, optional (default:(1.5,1.5))
            Parameters of prior distribution for change in SDMS (beta-distribution)

        random_state: int, optional (default:0)
            The state of random element
    '''

    def __init__(self, norm_file, max_clu=5, num_em=3, std_flg=True,
                 sdms_lam=1,sdms_beta=(1.5,1.5),random_state=0):
        '''constructor
        '''

        self.BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.DATA_DIR = os.path.normpath(os.path.join(self.BASE_DIR, './data'))
        self.OUT_DIR = os.path.normpath(os.path.join(self.DATA_DIR, './out'))
        self.norm_file = os.path.normpath(os.path.join(self.DATA_DIR, './normfile/'+norm_file))

        ### parameter for sample data
        self.max_clu = max_clu
        self.num_em = num_em
        self.std_flg = std_flg
        self.sdms_lam = sdms_lam
        self.sdms_beta = sdms_beta
        self.random_state = random_state
        self.rand_obj = np.random.RandomState(seed=self.random_state)

        ### internal parameter
        self.criterion = 'NML'

        self.params = {'max_clu':self.max_clu,'num_em':self.num_em}

    def parameter_check(self,data):
        '''Parameter check of NML
            data: array, shape ( time, ) (# of data, dimension)
                The target dataset
        '''
        norm_NML = self.read_norm_NML()
        N = 0
        m = 0
        for i,d in enumerate(data):
            N = np.max([N,len(d)])
            m = np.max([m,len(d[0])])

        ### check dimension
        str_assert_dim = 'Dimension specification is incorrect [ self:%d, norm_file:%d ]'%(m,norm_NML['param']['dim'])
        assert m == norm_NML['param']['dim'], str_assert_dim

        ### check the number of data
        str_assert_num = 'The number of data specification is incorrect [ self:%d, norm_file:%d ]'%(N,norm_NML['param']['num_data'])
        assert N <= norm_NML['param']['num_data'], str_assert_num

        ### check the number of cluster
        str_assert_clu = 'The number of cluster specification is incorrect [ self:%d, norm_file:%d ]'%(self.max_clu,norm_NML['param']['num_cluster'])
        assert self.max_clu <= norm_NML['param']['num_cluster'], str_assert_clu

    def read_norm_NML(self):
        '''read file which has normalization term of NML
        '''
        f = open(self.norm_file, 'r')
        jsonData = json.load(f)
        f.close()

        self.norm_nml = jsonData

        return jsonData

    def fit(self, data, progress=False):
        '''Fit to the dataset
        [Parameters]
            data: array, shape ( time, # of data, dimension)
                The target dataset

            progress: boolean, optional (default:False)
                Wheter to print the progress or not
        '''
        self.parameter_check(data)
        self.original_data = data
        self.calc_ts_nml(data,progress=progress)

    def _calc_nml_cluster(self, data):
        '''Calculate NML for each number of clusters using information criterion with EM algorithm
        [Parameters]
            data: array, shape ( # of data, dimension)
                The target dataset
        '''

        current_cri = 10**100
        optimal_z = []
        optimal_model = []
        optimal_criterion = 0
        optimal_num_clu = 0

        j_nml = self.read_norm_NML()
        init_clu = [0 for _ in range(data.shape[0])]
        init_num_clu = len(set(init_clu))

        ### standerization
        if self.std_flg:
            std_data = sp.stats.zscore(data, axis=0)
        else:
            std_data = data

        cri_list = pd.DataFrame(np.zeros((self.max_clu,1)),columns=[self.criterion])
        for k in range(1,self.max_clu+1):
            clu_init = copy.copy(np.array(init_clu))
            if k < init_num_clu:
                we_bool = clu_init>=k
                clu_init[we_bool] = [np.int64(self.rand_obj.random_sample()*k) for _ in range(np.sum(we_bool))]
            elif k == init_num_clu:
                clu_init = clu_init
            elif k > init_num_clu:
                p = ( k - init_num_clu ) / k
                we_list = [i for i in range(clu_init.shape[0]) if self.rand_obj.random_sample()<p ]
                clu_init[we_list] = [np.int64(self.rand_obj.random_sample()*k) for _ in range(len(we_list))]

            em_cri = np.inf
            z_tmp = np.array([])
            ### simulate with EM algorithm
            for _i in range(self.num_em):
                gmm_model = GaussianMixture(n_components=k,random_state=self.random_state+_i)
                gmm_model.fit(std_data)
                z = gmm_model.predict(std_data)
                cnml = cNML.calcGMMNML()
                log_likelihood = cnml.calc_log_likelihood(X=std_data,Z=z)

                ### read normalization term
                norm_cri = j_nml['normNML']['cluster_'+str(k)][data.shape[0]]
                nml_tmp = self.calc_criterion(log_likelihood=log_likelihood, norm_cri=norm_cri)


                if _i == 0:
                    cri_list.ix[k-1,self.criterion] = nml_tmp
                    z_tmp = z
                    em_cri = nml_tmp
                    final_model = gmm_model
                if em_cri > nml_tmp:
                    cri_list.ix[k-1,self.criterion] = nml_tmp
                    z_tmp = z
                    em_cri = nml_tmp
                    final_model = gmm_model

            optimal_z.append(z_tmp.tolist())
            optimal_model.append(final_model)

        return cri_list, optimal_z, optimal_model

    def calc_ts_nml(self,data,progress=False):
        '''Calculate NML for each number of clusters using information criterion with EM algorithm
        [Parameters]
            data: array, shape ( time, # of data, dimension)
                The target dataset

            progress: boolean, optional (default:False)
                Wheter to print the progress or not
        '''
        j_nml = self.read_norm_NML()

        st = time.time()
        nml_res = []
        opt_clu_list = []
        change_count = 0
        tmp_opt_cluster = None
        dict_result = {}
        ts_clu_index = []
        init_time = 0
        for t in range(init_time,data.shape[0]):
            tar_data = np.array(copy.copy(data[t]))

            if t == init_time:
                cri_list, optimal_z, optimal_model = self._calc_nml_cluster(data=tar_data)
            else:
                cri_list, optimal_z, optimal_model = self._calc_nml_cluster(data=tar_data)

            if t == init_time:
                tmp_opt_cluster = np.nanargmin(cri_list[self.criterion]) + 1
                opt_clu_list.append(tmp_opt_cluster)
                nml_res.append(list(cri_list[self.criterion]))

            else:
                t_nml = cri_list[self.criterion].values
                tmp_opt_cluster = np.nanargmin(cri_list[self.criterion]) + 1
                nml_res.append(t_nml.tolist())

            ### Dynamic Model Selection
            dms_obj = dms.dynamicModelSelection(sdms_lam=self.sdms_lam,sdms_beta=self.sdms_beta)
            dms_obj.calc_simple_sdms(np.array(nml_res).T)
            dms_opt_k = dms_obj.l_opt_k
            l_code_length = dms_obj.l_code_length

            curr_model = optimal_model[dms_opt_k[-1]-1]
            ts_clu_index.append(optimal_z[dms_opt_k[-1]-1])

            if progress:
                elapsed_time = time.time() - st
                print( 'time series:%d, # of clusters:[ NML:%d, DMS:%d ], elapsed-time:%d[s]'
                        %(t,tmp_opt_cluster,dms_opt_k[-1],elapsed_time) )

        self.ts_nml = np.array(nml_res).T
        self.ts_clu_index = ts_clu_index

    def calc_criterion(self, log_likelihood, norm_cri):
        cri = - log_likelihood + norm_cri

        return cri
