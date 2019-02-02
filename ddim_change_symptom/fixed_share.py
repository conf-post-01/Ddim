# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
import pandas as pd
from sklearn.preprocessing import normalize

import os
import sys
import copy
import json
import matplotlib.pyplot as plt
from datetime import datetime as dt

path = os.path.join(os.path.dirname(__file__), '../')
sys.path.append(path)
import ddim_change_symptom.dynamic_model_selection as dms

class calcFS():
    '''Calculate Fixed Share algorithm
    [Parameters]
        sdms_lam: float, optional (default:1)
            Scale of change code-length in SDMS

        sdms_beta: float, optional (default:(1.5,1.5))
            Parameters of prior distribution for change in SDMS (beta-distribution)

        temp_beta: float, optional (default:1)
            The temperature parameter 'beta' in Ddim

        fs_alpha: float, optional (default:0.1)
            The weight parameter for fixed share algorithm (range:[0,1])
    '''
    def __init__(self, sdms_lam=1, sdms_beta=(1.5,1.5), temp_beta=1, fs_alpha=.1):
        '''constructor'''

        self.BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.DATA_DIR = os.path.normpath(os.path.join(self.BASE_DIR, './data'))
        self.OUT_DIR = os.path.normpath(os.path.join(self.DATA_DIR, './out'))

        ### SDMS parameters
        self.sdms_lam = sdms_lam
        self.sdms_beta = sdms_beta

        ### define parameter
        self.temp_beta = temp_beta
        self.fs_alpha = fs_alpha

        ### internal parameter
        self.cl_N = 3

        ### output
        self.l_bx_fs = None

    def calc_ts_of_opt_k(self,ts_nml):
        '''Calculate optimal cluster using SDMS algorithm
        [Parameters]
            ts_nml: array, shape ( max # of cluster, time)
                The target nml list
        '''
        dms_obj = dms.dynamicModelSelection(sdms_lam=self.sdms_lam,sdms_beta=self.sdms_beta)
        dms_obj.calc_simple_sdms(ts_nml)

        self.opt_k = dms_obj.l_opt_k
        self.sdms_code_length = dms_obj.l_code_length
        self.dms_obj = dms_obj

        return dms_obj.l_opt_k, dms_obj.l_code_length

    def _ts_nml_to_weight(self,ts_nml):
        '''Calclate probability with time-series nml data
            ts_nml: array, shape ( max # of cluster, time)
                The target nml list
        '''

        min_nml = np.nanmin(ts_nml, axis=0)
        ts_nml = ts_nml - min_nml
        exp_nml = np.exp(-ts_nml)
        exp_nml = np.nan_to_num(exp_nml)

        return exp_nml

    def _calc_best_expert_fs(self,ts_nml):
        '''Calclate best expert using fixed share algorithm
            ts_nml: array, shape ( max # of cluster, time)
                The target nml list
        '''
        target_nml = np.array(ts_nml)

        ### SDMS algorithm
        opt_k_list, sdms_code_length = self.calc_ts_of_opt_k(ts_nml=target_nml)

        model_weight = self._ts_nml_to_weight(target_nml * self.temp_beta)
        l_w = np.zeros(target_nml.shape)
        l_bx_fs = []
        l_bx_fs_weightsum = []
        K = target_nml.shape[0]
        for t in range(target_nml.shape[1]):
            if t == 0:
                l_w[:,t] = np.ones(K) / K
                l_bx_fs.append(opt_k_list[t])
                l_bx_fs_weightsum.append(np.nan)
            else:
                l_w_tmp = l_w[:,t-1] * model_weight[:,t-1]
                ch_mat = (1-self.fs_alpha)*np.eye(K) + self.fs_alpha/(K-1) * (np.ones((K,K))-np.eye(K))
                l_w[:,t] = ch_mat @ l_w_tmp
                l_bx_fs.append(np.nanargmax(l_w[:,t])+1)

                norm_weight = normalize(l_w[:,t].reshape(1,-1),norm='l1')[0]
                weightsum_tmp = np.sum( np.argsort(norm_weight)[::-1][:2] * normalize(np.sort(norm_weight)[::-1][:2].reshape(1,-1),norm='l1') ) + 1
                l_bx_fs_weightsum.append(weightsum_tmp)

        self.l_bx_fs = l_bx_fs
        self.l_bx_fs_weightsum = l_bx_fs_weightsum

        return l_bx_fs

    def calc_ts_of_FS(self,ts_nml):
        '''Calclate best expert using fixed share algorithm
            ts_nml: array, shape ( max # of cluster, time)
                The target nml list
        '''
        l_bx_fs = self._calc_best_expert_fs(ts_nml=ts_nml)

        return l_bx_fs
