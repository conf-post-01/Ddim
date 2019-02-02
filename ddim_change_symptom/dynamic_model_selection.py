# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
import pandas as pd
from sklearn.preprocessing import normalize

import os
import copy
import json
import matplotlib.pyplot as plt
from datetime import datetime as dt

class dynamicModelSelection():
    '''Calculate optimal number of cluster using SDMS algorithm
    [Parameters]
        sdms_lam: float, optional (default:1)
            Scale of change code-length in SDMS

        sdms_beta: float, optional (default:(1.5,1.5))
            Parameters of prior distribution for change in SDMS (beta-distribution)
    '''

    def __init__(self,sdms_lam=1,sdms_beta=(1.5,1.5)):
        '''constructor'''

        self.BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.DATA_DIR = os.path.normpath(os.path.join(self.BASE_DIR, './data'))
        self.OUT_DIR = os.path.normpath(os.path.join(self.DATA_DIR, './out'))

        ### input parameters
        self.sdms_lam = sdms_lam
        self.sdms_beta = sdms_beta

        ### output
        self.l_opt_k = None
        self.l_code_length = None
        self.c_list = None
        self.not_c_list = None
        self.alpha_list = None

    def calc_simple_sdms(self,target_nml):
        '''Calculate Dynamic Model Selection
        [Parameters]
            target_nml: array, shape ( max # of cluster, time)
                The target nml list
        '''

        beta_a = self.sdms_beta[0]
        beta_b = self.sdms_beta[1]
        l_opt_k = []
        l_code_length = []
        change_count = 0
        tmp_opt_cluster = None
        c_list = []
        not_c_list = []
        alpha_list = []
        for t in range(target_nml.shape[1]):
            if t == 0:
                t_nml = target_nml[:,t]
                tmp_opt_cluster = 1

            t_nml = target_nml[:,t]
            pre_num_cluster = copy.copy(tmp_opt_cluster)

            ### MAP estimation
            ### if beta_a=1.5 and beta_b=1.5, MAP estimation = Krichevsky and Trofimov estimation
            if t == 0 or t == 1:
                hat_alpha = 0.5
            else:
                hat_alpha = ( change_count + beta_a - 1 ) / ( t-1 + beta_a + beta_b - 2 )
                
            if (tmp_opt_cluster==1) or (tmp_opt_cluster==target_nml.shape[0]):
                not_change_code_length = -np.log(1.-hat_alpha/2.)
            else:
                not_change_code_length = -np.log(1.-hat_alpha)
            change_code_length = -np.log(hat_alpha/2.)

            #############
            ### SDMS
            cl_change = np.zeros(3)

            ### change to small
            if tmp_opt_cluster==1:
                cl_change[0] = np.inf
            else:
                cl_change[0] = t_nml[tmp_opt_cluster-1-1] + change_code_length * self.sdms_lam

            ### not change
            cl_change[1] = t_nml[tmp_opt_cluster-1] + not_change_code_length * self.sdms_lam

            ### change to big
            if tmp_opt_cluster==target_nml.shape[0]:
                cl_change[2] = np.inf
            else:
                cl_change[2] = t_nml[tmp_opt_cluster-1+1] + change_code_length * self.sdms_lam

            if np.isnan(cl_change).all():
                tmp_opt_cluster = tmp_opt_cluster
            else:
                tmp_opt_cluster += np.nanargmin(cl_change)-1

            l_opt_k.append(tmp_opt_cluster)
            l_code_length.append(cl_change.tolist())
            c_list.append(change_code_length)
            not_c_list.append(not_change_code_length)
            alpha_list.append(hat_alpha)
            #############

            if pre_num_cluster != tmp_opt_cluster:
                change_count += 1

        self.l_opt_k = l_opt_k
        self.l_code_length = l_code_length
        self.c_list = c_list
        self.not_c_list = not_c_list
        self.alpha_list = alpha_list
