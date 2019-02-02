# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

import argparse
import glob
import sys
import os
import sys
import random
import copy
import json
import time
from datetime import datetime as dt
import matplotlib.pyplot as plt

path = os.path.join(os.path.dirname(__file__), '../')
sys.path.append(path)
import ddim_change_symptom.gmm_generator as gg
import ddim_change_symptom.ts_clustering as tc
import ddim_change_symptom.dynamic_model_selection as dms
import ddim_change_symptom.descriptional_dimension as ddim
import ddim_change_symptom.symptom_evaluation as sym_eval
import ddim_change_symptom.fixed_share as fs

class simDdimRealData():
    '''Simulate Descriptional Dimension (Ddim)
    [Parameters]
        random_state: int, optional (default:0)
            The state of random element

        conf_file: string, optional (default:'ddim_paramset_single.json')
            The file of paramter set

        c_pattern: string, optional (default:'single')
            The gradual change point pattern

        temp_beta: float, optional (default:1)
            The temperature parameter 'beta' in Ddim

        out_flg: boolean, optional (default:False)
            Whether to output result or not

    '''

    def __init__(self, data_file, random_state=0, conf_file='ddim_paramset_realdata.json',
                 temp_beta=None, out_flg=False):

        now_time = dt.now()
        now_str = now_time.strftime("%Y%m%d-%H%M%S")
        date_str = now_time.strftime("%Y%m%d")

        ### params
        self.random_state = random_state
        self.conf_file = conf_file
        self.temp_beta = temp_beta
        self.out_flg = out_flg

        ### internal parameter
        self.data_column = 'amount_data'
        self.sim_title = 'ddim_realdata'
        self.date_str = date_str
        self.file_name = 'ddim_result'

        self.FILE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.OUT_DIR = os.path.normpath(os.path.join(self.FILE_DIR, './data/out/%s_%s'%(self.sim_title,date_str)))
        self.CONF_DIR = os.path.normpath(os.path.join(self.FILE_DIR, './conf'))
        self.out_data_file = '%s/%s_rand%d_%s.json'%(self.OUT_DIR,self.file_name,random_state,now_str)

        self.data_file = os.path.normpath(os.path.join(self.FILE_DIR, './data/%s'%(data_file)))

        if not os.path.isdir(self.OUT_DIR):
            os.mkdir(self.OUT_DIR)

        ### args
        self._read_conf()

        ### input dataset
        self._read_dataset()

        ### output
        self.out = {}

    def _read_conf(self):
        '''Read conf file
        '''
        conf_file = self.CONF_DIR + '/' + self.conf_file

        ### read json
        f = open(conf_file, 'r')
        params = json.load(f)
        f.close()

        self.params = params

    def _read_dataset(self):
        '''Read given dataset
        '''
        ### read json
        f = open(self.data_file, 'r')
        j_data = json.load(f)
        f.close()

        amount_dataset = np.array(j_data[self.data_column])

        self.dataset = amount_dataset


    def simulate_algorithm_to_dataset(self):
        '''Simulate algorithm using given dataset
        '''

        ### parameter of simulation
        norm_file = self.params['norm_file']
        iter_count = self.params['simulation']['iter']

        ### parameter of clustering
        num_em = self.params['clustering_params']['num_em']
        max_clu = self.params['clustering_params']['max_clu']

        ### parameter of SDMS
        sdms_lam = self.params['sdms_params']['sdms_lam']
        sdms_beta = self.params['sdms_params']['sdms_beta']

        ### parameter of Ddim
        temp_beta_unit = self.params['sim_params']['temp_beta_unit']
        th_change_diff = self.params['sim_params']['th_change_diff']
        th_change_th = self.params['sim_params']['th_change_th']
        th_change_sdms = self.params['sim_params']['th_change_sdms']
        th_change_fs = self.params['sim_params']['th_change_fs']
        th_change_ws_fs = self.params['sim_params']['th_change_ws_fs']
        th_change_ws_fs_th = self.params['sim_params']['th_change_ws_fs_th']

        ### parameter of fixed share
        fs_alpha = self.params['sim_params']['fs_alpha']
        fs_temp_beta = self.params['sim_params']['fs_temp_beta']

        # num = self.dataset.shape[1]
        if self.temp_beta is None:
            # temp_beta = temp_beta_unit / np.sqrt(num)
            temp_beta = 0.05
        else:
            temp_beta = self.temp_beta

        st = time.time()
        res_opt_k = np.zeros((iter_count,self.dataset.shape[0]))
        res_ent = np.zeros((iter_count,self.dataset.shape[0]))
        res_ddim = np.zeros((iter_count,self.dataset.shape[0]))
        res_fs = np.zeros((iter_count,self.dataset.shape[0]))
        res_ws_fs = np.zeros((iter_count,self.dataset.shape[0]))
        for i in range(iter_count):
            tc_obj = tc.gmmSeqClustering(max_clu=max_clu,num_em=num_em,norm_file=norm_file,sdms_lam=sdms_lam,sdms_beta=sdms_beta)
            tc_obj.fit(self.dataset,progress=True)

            dms_obj = dms.dynamicModelSelection(sdms_lam=sdms_lam,sdms_beta=sdms_beta)
            dms_obj.calc_simple_sdms(tc_obj.ts_nml)

            ### calculate Descriptional Dimension
            ddim_obj = ddim.calcDdim(sdms_lam=sdms_lam,sdms_beta=sdms_beta,temp_beta=temp_beta)
            ddim_value = ddim_obj.calc_ts_of_Ddim(tc_obj.ts_nml)

            ### fixed share
            fs_obj = fs.calcFS(sdms_lam=sdms_lam, sdms_beta=sdms_beta, temp_beta=fs_temp_beta, fs_alpha=fs_alpha)
            fs_obj.calc_ts_of_FS(tc_obj.ts_nml)

            res_opt_k[i,:] = dms_obj.l_opt_k
            res_ddim[i,:] = ddim_value
            res_fs[i,:] = fs_obj.l_bx_fs
            res_ws_fs[i,:] = fs_obj.l_bx_fs_weightsum

            elapsed_time = time.time() - st
            print('=== optimal k (iter:%d,elapsed-time:%d[s]) ==='%(i,elapsed_time))
            print(dms_obj.l_opt_k)

        ### evaluation
        eval_obj = sym_eval.evalSymptom()

        self.tc_obj = tc_obj

        avg_opt_k = np.mean(res_opt_k,axis=0)
        avg_ddim = np.mean(res_ddim,axis=0)
        avg_fs = np.mean(res_fs,axis=0)
        avg_ws_fs = np.mean(res_ws_fs,axis=0)

        ### symptom detection: Diff
        t_criterion_diff = [np.nan]
        t_criterion_diff.extend( ( np.diff(avg_ddim) ).tolist())
        t_criterion_diff = np.array(t_criterion_diff)
        t_criterion_diff = np.abs(t_criterion_diff)
        sy_diff = eval_obj.symptom_detection(t_criterion_diff,th_change=th_change_diff)

        ### symptom detection: Threshold
        t_criterion_th = np.absolute( avg_opt_k - avg_ddim )
        sy_th = eval_obj.symptom_detection(t_criterion_th,th_change=th_change_th)

        ### SDMS
        t_criterion_sdms = [np.nan]
        t_criterion_sdms.extend( ( np.diff(avg_opt_k) ).tolist())
        t_criterion_sdms = np.array(t_criterion_sdms)
        sy_sdms = eval_obj.symptom_detection(t_criterion_sdms,th_change=th_change_sdms)

        ### fixed share
        t_criterion_fs = [np.nan]
        t_criterion_fs.extend( ( np.diff(avg_fs) ).tolist())
        t_criterion_fs = np.array(t_criterion_fs)
        sy_fs = eval_obj.symptom_detection(t_criterion_fs,th_change=th_change_fs)

        ### weighted average fixed share: Diff
        t_criterion_ws_fs = [np.nan]
        t_criterion_ws_fs.extend( ( np.diff(avg_ws_fs) ).tolist())
        t_criterion_ws_fs = np.array(t_criterion_ws_fs)
        sy_ws_fs = eval_obj.symptom_detection(t_criterion_ws_fs,th_change=th_change_ws_fs)

        ### weighted average fixed share: Threshold
        t_criterion_ws_fs_th = np.absolute( avg_opt_k - avg_ws_fs )
        sy_ws_fs_th = eval_obj.symptom_detection(t_criterion_ws_fs_th,th_change=th_change_ws_fs_th)

        self.out['params'] = self.params
        self.out['params']['file'] = self.conf_file
        self.out['params']['temp_beta'] = self.temp_beta
        self.out['result'] = {}
        self.out['result']['opt_k'] = res_opt_k.tolist()
        self.out['result']['ddim'] = res_ddim.tolist()
        self.out['result']['avg_opt_k'] = avg_opt_k.tolist()
        self.out['result']['avg_ddim'] = avg_ddim.tolist()
        self.out['result']['avg_fs'] = avg_fs.tolist()
        self.out['result']['avg_ws_fs'] = avg_ws_fs.tolist()
        self.out['result']['symptom_diff'] = sy_diff
        self.out['result']['symptom_th'] = sy_th
        self.out['result']['symptom_sdms'] = sy_sdms
        self.out['result']['symptom_fs'] = sy_fs
        self.out['result']['symptom_ws_fs'] = sy_ws_fs

        self.out['result']['evaluation'] = {}
        ### Diff
        self.out['result']['evaluation']['t_criterion_diff'] = t_criterion_diff.tolist()
        ### TH
        self.out['result']['evaluation']['t_criterion_th'] = t_criterion_th.tolist()
        ### SDMS
        self.out['result']['evaluation']['t_criterion_sdms'] = t_criterion_sdms.tolist()
        ### FS
        self.out['result']['evaluation']['t_criterion_fs'] = t_criterion_fs.tolist()
        ### Weighted average FS: Diff
        self.out['result']['evaluation']['t_criterion_ws_fs'] = t_criterion_ws_fs.tolist()
        ### Weighted average FS: TH
        self.out['result']['evaluation']['t_criterion_ws_fs_th'] = t_criterion_ws_fs_th.tolist()


        if self.out_flg:
            f = open(self.out_data_file, 'w')
            json.dump(self.out, f, indent=4)
            f.close()

    def create_graph(self,initial_period=0,ts_ind=None,c_ind_th=None,c_ind_diff=None,fs_flg=False,tofile=False,file_format='eps'):

        '''Create graph with result of Ddim
        [Parameters]
            tofile: boolean, optional (default:False)
                Whether to out a file or not

            file_format: string, optional (default:'eps')
                Output file type (Usually 'eps' or 'png')
        '''

        opt_k = np.array(self.out['result']['avg_opt_k'])
        l_ddim = np.array(self.out['result']['avg_ddim'])
        avg_ws_fs = np.array(self.out['result']['avg_ws_fs'])

        ### Result of Diff
        t_criterion_diff = self.out['result']['evaluation']['t_criterion_diff']
        sy_diff = self.out['result']['symptom_diff']

        ### Result of TH
        t_criterion_th = self.out['result']['evaluation']['t_criterion_th']
        sy_th = self.out['result']['symptom_th']

        ### SDMS
        sy_sdms = self.out['result']['symptom_sdms']

        sy_th_move = ( np.array(sy_th) + initial_period ).tolist()
        sy_diff_move = ( np.array(sy_diff) + initial_period ).tolist()
        sy_sdms_move = ( np.array(sy_sdms) + initial_period ).tolist()

        ### Define directory name and file name
        now_time = dt.now()
        now_str = now_time.strftime("%Y%m%d-%H%M%S")
        date_str = now_time.strftime("%Y%m%d")

        dir_name = self.OUT_DIR
        if not os.path.isdir(dir_name):
            os.mkdir(dir_name)

        file_name = '%s/gmm_clu_ddim_realdata'%(dir_name)
        out_fig = '%s_%s.%s'%(file_name,now_str,file_format)

        max_time = opt_k.shape[0]
        ts = np.array([i for i in range(max_time)])
        if ts_ind is None:
            ts_ind = ts
        if c_ind_th is None:
            c_ind_th = sy_th
        if c_ind_diff is None:
            c_ind_diff = sy_diff

        ts_sdms, opt_k_sdms, ts_ind_sdms = self._create_sequence(ts,ts_ind,opt_k)

        alpha_true = 1.
        l_width = 5.
        ddim_width = 12.

        fig = plt.figure(figsize=(20,12))
        plt.rcParams["font.size"] = 24
        ax = fig.add_subplot(111)

        title_str = 'Ddim for Gradual Change \n<Symptom> TH:%s, Diff:%s, SDMS:%s' \
                                %(str(sy_th_move), str(sy_diff_move), str(sy_sdms_move) )
        ax.plot(ts_sdms[ts_ind_sdms]+initial_period,opt_k_sdms[ts_ind_sdms],'b-',linewidth=l_width,label='SDMS')
        if fs_flg:
            ax.plot(ts[ts_ind]+initial_period,avg_ws_fs[ts_ind],'c-',linewidth=l_width,label='WA-FS')
        avg_ws_fs
        ax.plot(ts[ts_ind]+initial_period,l_ddim[ts_ind],'g-',linewidth=ddim_width, label='Ddim')
        plt.ylabel('SDMS / Ddim')
        plt.xlabel('Time')

        ax2 = ax.twinx()

        cri_min, cri_max = 0, 1.5
        ax2.plot(ts[ts_ind]+initial_period,np.array(t_criterion_th)[ts_ind],'r--',linewidth=l_width, label='TH')
        for c in c_ind_th:
            ax2.plot([c+initial_period,c+initial_period],[cri_min,cri_max],'r--',linewidth=l_width)
        ax2.plot(ts[ts_ind]+initial_period,np.array(t_criterion_diff)[ts_ind],'m-.',linewidth=l_width,label='Diff')
        for c in c_ind_diff:
            ax2.plot([c+initial_period,c+initial_period],[cri_min,cri_max],'m-.',linewidth=l_width)
        ax.legend(loc=2)
        ax2.legend(loc=4)
        plt.ylabel('TH / Diff')
        # plt.title(title_str)
        plt.xlabel('time')
        if tofile:
            plt.savefig(out_fig, format=file_format)
            plt.show()
        else:
            plt.show()

    def _create_sequence(self,ts,ts_ind,opt_k):
        ts_out = []
        opt_k_out = []
        ts_ind_out = []
        curr_o = None
        curr_lag = 0
        for i,(t,o) in enumerate(zip(ts,opt_k)):
            if i == 0:
                ts_out.append(t)
                opt_k_out.append(o)
                if t in ts_ind:
                    ts_ind_out.append(t + curr_lag)
                curr_o = o
                continue

            if curr_o == o:
                ts_out.append(t)
                opt_k_out.append(o)
                if t in ts_ind:
                    ts_ind_out.append(t + curr_lag)
                curr_o = o
            else:
                ts_out.append(t)
                ts_out.append(t)
                opt_k_out.append(curr_o)
                opt_k_out.append(o)
                if t in ts_ind:
                    ts_ind_out.append(t + curr_lag)
                    ts_ind_out.append(t + curr_lag + 1)
                curr_lag += 1
                curr_o = o
        ts_out = np.array(ts_out)
        opt_k_out = np.array(opt_k_out)

        return ts_out, opt_k_out, ts_ind_out
