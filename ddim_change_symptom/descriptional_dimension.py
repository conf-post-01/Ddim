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

class calcDdim():
    '''Calculate Descriptional Dimension (Ddim)
    [Parameters]
        sdms_lam: float, optional (default:1)
            Scale of change code-length in SDMS

        sdms_beta: float, optional (default:(1.5,1.5))
            Parameters of prior distribution for change in SDMS (beta-distribution)

        temp_beta: float, optional (default:1)
            The temperature parameter 'beta' in Ddim

        num_minimum: int, optional (default:2)
            The number of model classes used for calculating Ddim

    [Output]
        l_ddim: list, shape: (The length of sequence)
            The calculated Ddim sequence

    '''

    def __init__(self, sdms_lam=1, sdms_beta=(1.5,1.5),
                 temp_beta=1, num_minimum=2):
        '''constructor
        '''

        self.BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.DATA_DIR = os.path.normpath(os.path.join(self.BASE_DIR, './data'))
        self.OUT_DIR = os.path.normpath(os.path.join(self.DATA_DIR, './out'))

        ### SDMS parameters
        self.sdms_lam = sdms_lam
        self.sdms_beta = sdms_beta

        ### Ddim parameters
        self.temp_beta = temp_beta
        self.num_minimum = num_minimum

        ### output
        self.l_ddim = None

    def calc_ts_of_opt_k(self,ts_nml):
        '''Calculate optimal model class sequence using SDMS
        [Parameters]
            ts_nml: np.array, shape ( num_cluster, time )
                The sequence list of code-length
        '''
        dms_obj = dms.dynamicModelSelection(sdms_lam=self.sdms_lam,sdms_beta=self.sdms_beta)
        dms_obj.calc_simple_sdms(ts_nml)

        self.opt_k = dms_obj.l_opt_k
        self.sdms_code_length = dms_obj.l_code_length
        self.dms_obj = dms_obj

        return dms_obj.l_opt_k, dms_obj.l_code_length

    def _ts_nml_to_probability(self,ts_nml):
        '''Calclate probability with time-series nml data
        [Parameters]
            ts_nml: np.array, shape ( num_cluster, time )
                The sequence list of code-length
        '''

        ### Change scale
        min_nml = np.nanmin(ts_nml, axis=0)
        ts_nml = ts_nml - min_nml
        exp_nml = np.exp(-ts_nml)
        exp_nml = np.nan_to_num(exp_nml)
        exp_norm_nml = normalize(exp_nml, norm='l1', axis=0)

        return exp_norm_nml

    def _calc_ddim_fusion(self,ts_nml):
        '''Calculating Descriptional Dimension (Fusion)
        [Parameters]
            ts_nml: np.array, shape ( num_cluster, time )
                The sequence list of code-length
        '''
        target_nml = np.array(ts_nml)

        ### SDMS
        opt_k_list, sdms_code_length = self.calc_ts_of_opt_k(ts_nml=target_nml)

        ### calculating method with SDMS
        t_clu = []
        for t,sdms_nml in enumerate(sdms_code_length):
            if np.isnan(sdms_nml).all():
                arg_sdms = t_clu[-1][1]
            else:
                arg_sdms = np.nanargmin(sdms_nml)
            tm_t_clu = list(range(opt_k_list[t]-arg_sdms,opt_k_list[t]-arg_sdms+3))
            t_clu.append(tm_t_clu)

        ### calculate probability of each model
        model_prob = self._ts_nml_to_probability(np.array(sdms_code_length).T * self.temp_beta)
        l_ddim = np.sum(np.array(t_clu).T * model_prob,axis=0)
        l_bool = np.sum(model_prob,axis=0)==0
        l_ddim[l_bool] = np.nan
        l_ddim = l_ddim.tolist()

        return l_ddim

    def calc_ts_of_Ddim(self,ts_nml,m='Fusion'):
        if m == 'Fusion':
            l_ddim = self._calc_ddim_fusion(ts_nml=ts_nml)
            self.l_ddim = l_ddim
        else:
            l_ddim = None
            self.l_ddim = l_ddim

        return l_ddim
