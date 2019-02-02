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

class simDdim():
    '''Simulate Descriptive Dimension (Ddim)
    [Parameters]
        random_state: int, optional (default:0)
            The state of random element.

        conf_file: string, optional (default:'ddim_paramset_single.json')
            The file of paramter set.

        c_pattern: string, optional (default:'single')
            The gradual change pattern specifying single change point or multi- ple ones.

        temp_beta: float, optional (default:None)
            The temperature parameter 'beta' in Ddim.
            If None, the temperature parameter is 1/sqrt(n).

        num_data: int, optional (default:1000)
            The number of dataset at each time.

        alpha: float, optional (default:1.0)
            The defined change type alpha.
            If 1.0, the change pattern in transition period is linear change.

        out_flg: boolean, optional (default:False)
            Whether to output result with json file or not.

    '''

    def __init__(self, random_state=0, conf_file='ddim_paramset_single.json', c_pattern='single',
                 temp_beta=None, num_data=1000, alpha=1.0, out_flg=False):

        now_time = dt.now()
        now_str = now_time.strftime("%Y%m%d-%H%M%S")
        date_str = now_time.strftime("%Y%m%d")

        ### params
        self.random_state = random_state
        self.conf_file = conf_file
        self.c_pattern = c_pattern
        self.temp_beta = temp_beta
        self.num_data = num_data
        self.alpha = alpha
        self.out_flg = out_flg

        ### internal parameter
        self.sim_title = 'ddim'
        self.date_str = date_str
        self.file_name = 'ddim_result'

        self.FILE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.OUT_DIR = os.path.normpath(os.path.join(self.FILE_DIR, './data/out/%s_%s_%s'%(self.sim_title,self.c_pattern,date_str)))
        self.CONF_DIR = os.path.normpath(os.path.join(self.FILE_DIR, './conf'))
        self.out_data_file = '%s/%s_rand%d_alpha%.2f_%s.json'%(self.OUT_DIR,self.file_name,random_state,alpha,now_str)

        if not os.path.isdir(self.OUT_DIR):
            os.mkdir(self.OUT_DIR)

        ### args
        self.read_conf()

        ### output
        self.out = {}

    def read_conf(self):
        '''Read conf file
        '''
        conf_file = self.CONF_DIR + '/' + self.conf_file

        ### read json
        f = open(conf_file, 'r')
        params = json.load(f)
        f.close()

        self.params = params

    def _generate_simulation_dataset(self):
        '''Generate dataset
        '''
        if self.c_pattern == 'single':
            ### parameter of data generation
            l_T = self.params['data_params']['l_T']
            l_K = self.params['data_params']['l_K']
            l_mu = self.params['data_params']['l_mu']
            var = self.params['data_params']['var']
            corr = self.params['data_params']['corr']
            abs_mu = self.params['data_params']['abs_mu']

            ### generate dataset
            gg_obj = gg.gmmGenerator(num=self.num_data,l_T=l_T,l_K=l_K,
                                     l_mu=l_mu,var=var, corr=corr, abs_mu=abs_mu,
                                     random_state=self.random_state,change_alpha=self.alpha)
            data, _ = gg_obj.generate()

            true_k = gg_obj.true_k

        elif self.c_pattern == 'multi':
            ### parameter of data generation
            l_T = self.params['data_params']['l_T']
            l_K = self.params['data_params']['l_K']
            l_mu = self.params['data_params']['l_mu']
            var = self.params['data_params']['var']
            corr = self.params['data_params']['corr']
            abs_mu = self.params['data_params']['abs_mu']

            ### parameter of data generation (multi)
            l_T_m = self.params['data_params_multi']['l_T']
            l_K_m = self.params['data_params_multi']['l_K']
            l_mu_m = self.params['data_params_multi']['l_mu']
            var_m = self.params['data_params_multi']['var']
            corr_m = self.params['data_params_multi']['corr']
            abs_mu_m = self.params['data_params_multi']['abs_mu']

            ### generate dataset
            gg_obj_bef = gg.gmmGenerator(num=self.num_data,l_T=l_T,l_K=l_K,
                                         l_mu=l_mu,var=var, corr=corr, abs_mu=abs_mu,
                                         random_state=self.random_state,change_alpha=self.alpha)
            gg_obj_af = gg.gmmGenerator(num=self.num_data,l_T=l_T_m,l_K=l_K_m,
                                        l_mu=l_mu_m,var=var_m, corr=corr_m, abs_mu=abs_mu_m,
                                        random_state=self.random_state,change_alpha=self.alpha)

            data_bef, ind_bef = gg_obj_bef.generate(init_index=None)
            data_af, _ = gg_obj_af.generate(init_index=ind_bef[-1])
            data = np.concatenate((data_bef,data_af),axis=0)

            true_k = gg_obj_bef.true_k + gg_obj_af.true_k

        return data, true_k

    def _create_true_symptom(self):
        '''Create true symptom
        '''
        if self.c_pattern == 'single':
            l_T = self.params['data_params']['l_T']
            grad_range = [[l_T[0],l_T[0]+l_T[1]-1]]
        elif self.c_pattern == 'multi':
            l_T = self.params['data_params']['l_T']
            l_T_m = self.params['data_params_multi']['l_T']
            grad_range = [[l_T[0],l_T[0]+l_T[1]-1],[np.sum(l_T)+l_T_m[0],np.sum(l_T)+l_T_m[0]+l_T_m[1]-1]]

        return grad_range

    def simulate_algorithm(self):
        '''Simulate algorithm
        '''

        ### parameter of simulation
        norm_file = self.params['norm_file']
        iter_count = self.params['simulation']['iter']

        ### parameter of clustering
        num_em = self.params['clustering_params']['num_em']

        ### parameter of SDMS
        sdms_lam = self.params['sdms_params']['sdms_lam']
        sdms_beta = self.params['sdms_params']['sdms_beta']

        ### parameter of Ddim
        th_change_diff = self.params['sim_params']['th_change_diff']
        th_change_th = self.params['sim_params']['th_change_th']

        ### parameter of several models
        th_change_sdms = self.params['sim_params']['th_change_sdms']
        th_change_fs = self.params['sim_params']['th_change_fs']
        th_change_ws_fs = self.params['sim_params']['th_change_ws_fs']
        th_change_ws_fs_th = self.params['sim_params']['th_change_ws_fs_th']
        ben_T = self.params['sim_params']['ben_T']

        ### parameter of fixed share
        fs_alpha = self.params['sim_params']['fs_alpha']
        fs_temp_beta = self.params['sim_params']['fs_temp_beta']

        ### generate dataset
        data, true_k = self._generate_simulation_dataset()

        ### define scale parameter
        num = data.shape[1]
        if self.temp_beta is None:
            temp_beta = 1 / np.sqrt(num)
        else:
            temp_beta = self.temp_beta

        st = time.time()
        res_opt_k = np.zeros((iter_count,data.shape[0]))
        res_ddim = np.zeros((iter_count,data.shape[0]))
        res_fs = np.zeros((iter_count,data.shape[0]))
        res_ws_fs = np.zeros((iter_count,data.shape[0]))
        for i in range(iter_count):

            ### fit with original dataset
            tc_obj = tc.gmmSeqClustering(norm_file=norm_file,num_em=num_em,sdms_lam=sdms_lam,sdms_beta=sdms_beta,random_state=self.random_state)
            tc_obj.fit(data)

            ### dynamic model selection
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

        avg_opt_k = np.mean(res_opt_k,axis=0)
        avg_ddim = np.mean(res_ddim,axis=0)
        avg_fs = np.mean(res_fs,axis=0)
        avg_ws_fs = np.mean(res_ws_fs,axis=0)

        ### evaluation
        eval_obj = sym_eval.evalSymptom()
        grad_range = self._create_true_symptom()

        ### symptom detection: Diff
        t_criterion_diff = [np.nan]
        t_criterion_diff.extend( ( np.diff(avg_ddim) ).tolist())
        t_criterion_diff = np.array(t_criterion_diff)
        t_criterion_diff = np.abs(t_criterion_diff)
        sy_diff = eval_obj.symptom_detection(t_criterion_diff,th_change=th_change_diff)
        delay_value_diff, benefit_value_diff = eval_obj.eval_symptom(grad_range=grad_range,symptom=sy_diff,ben_T=ben_T)

        ### symptom detection: Threshold
        t_criterion_th = np.absolute( avg_opt_k - avg_ddim )
        sy_th = eval_obj.symptom_detection(t_criterion_th,th_change=th_change_th)
        delay_value_th, benefit_value_th = eval_obj.eval_symptom(grad_range=grad_range,symptom=sy_th,ben_T=ben_T)

        ### SDMS
        t_criterion_sdms = [np.nan]
        t_criterion_sdms.extend( ( np.diff(avg_opt_k) ).tolist())
        t_criterion_sdms = np.array(t_criterion_sdms)
        sy_sdms = eval_obj.symptom_detection(t_criterion_sdms,th_change=th_change_sdms)
        delay_value_sdms, benefit_value_sdms = eval_obj.eval_symptom(grad_range=grad_range,symptom=sy_sdms,ben_T=ben_T)

        ### fixed share
        t_criterion_fs = [np.nan]
        t_criterion_fs.extend( ( np.diff(avg_fs) ).tolist())
        t_criterion_fs = np.array(t_criterion_fs)
        sy_fs = eval_obj.symptom_detection(t_criterion_fs,th_change=th_change_fs)
        delay_value_fs, benefit_value_fs = eval_obj.eval_symptom(grad_range=grad_range,symptom=sy_fs,ben_T=ben_T)

        ### weighted average fixed share: Diff
        t_criterion_ws_fs = [np.nan]
        t_criterion_ws_fs.extend( ( np.diff(avg_ws_fs) ).tolist())
        t_criterion_ws_fs = np.array(t_criterion_ws_fs)
        sy_ws_fs = eval_obj.symptom_detection(t_criterion_ws_fs,th_change=th_change_ws_fs)
        delay_value_ws_fs, benefit_value_ws_fs = eval_obj.eval_symptom(grad_range=grad_range,symptom=sy_ws_fs,ben_T=ben_T)

        ### weighted average fixed share: Threshold
        t_criterion_ws_fs_th = np.absolute( avg_opt_k - avg_ws_fs )
        sy_ws_fs_th = eval_obj.symptom_detection(t_criterion_ws_fs_th,th_change=th_change_ws_fs_th)
        delay_value_ws_fs_th, benefit_value_ws_fs_th = eval_obj.eval_symptom(grad_range=grad_range,symptom=sy_ws_fs_th,ben_T=ben_T)


        self.params['temp_beta'] = temp_beta

        self.out['params'] = self.params
        self.out['params']['true_k'] = true_k
        self.out['params']['file'] = self.conf_file
        self.out['params']['pattern'] = self.c_pattern
        self.out['params']['temp_beta'] = self.temp_beta
        self.out['params']['n'] = self.num_data
        self.out['params']['alpha'] = self.alpha
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
        self.out['result']['evaluation']['delay_diff'] = delay_value_diff
        self.out['result']['evaluation']['benefit_diff'] = benefit_value_diff
        self.out['result']['evaluation']['t_criterion_diff'] = t_criterion_diff.tolist()
        ### TH
        self.out['result']['evaluation']['delay_th'] = delay_value_th
        self.out['result']['evaluation']['benefit_th'] = benefit_value_th
        self.out['result']['evaluation']['t_criterion_th'] = t_criterion_th.tolist()
        ### SDMS
        self.out['result']['evaluation']['delay_sdms'] = delay_value_sdms
        self.out['result']['evaluation']['benefit_sdms'] = benefit_value_sdms
        self.out['result']['evaluation']['t_criterion_sdms'] = t_criterion_sdms.tolist()
        ### FS
        self.out['result']['evaluation']['delay_fs'] = delay_value_fs
        self.out['result']['evaluation']['benefit_fs'] = benefit_value_fs
        self.out['result']['evaluation']['t_criterion_fs'] = t_criterion_fs.tolist()
        ### Weighted average FS: Diff
        self.out['result']['evaluation']['delay_ws_fs'] = delay_value_ws_fs
        self.out['result']['evaluation']['benefit_ws_fs'] = benefit_value_ws_fs
        self.out['result']['evaluation']['t_criterion_ws_fs'] = t_criterion_ws_fs.tolist()
        ### Weighted average FS: TH
        self.out['result']['evaluation']['delay_ws_fs_th'] = delay_value_ws_fs_th
        self.out['result']['evaluation']['benefit_ws_fs_th'] = benefit_value_ws_fs_th
        self.out['result']['evaluation']['t_criterion_ws_fs_th'] = t_criterion_ws_fs_th.tolist()

        if self.out_flg:
            f = open(self.out_data_file, 'w')
            json.dump(self.out, f, indent=4)
            f.close()

    def _calc_trans_period(self,true_k):
        '''Create graph with result of Ddim
        [Parameters]
            true_k: list
                List of true number of clusters at each time
        '''
        trans_index = np.array([i for i in range(len(true_k))])[np.isnan(true_k)]
        trans_period = []
        trans_tmp = [np.nan,np.nan]
        n_t = 0
        cont_flg = False
        for i,t in enumerate(trans_index):
            if not cont_flg:
                cont_flg = True
                trans_tmp[0] = t
                n_t = t
                continue

            if t != n_t+1 or i == len(trans_index)-1:
                if t != n_t+1:
                    trans_tmp[1] = n_t
                elif i == len(trans_index)-1:
                    trans_tmp[1] = t
                trans_period.append(trans_tmp)
                trans_tmp = [np.nan,np.nan]
                trans_tmp[0] = t
            n_t = t

        return trans_period

    def create_graph(self,type_name,tofile=False,file_format='eps'):
        '''Create graph with result of Ddim
        [Parameters]
            type_name: string
                Define file name ('single' or 'multi')

            tofile: boolean, optional (default:False)
                Whether to out a file or not

            file_format: string, optional (default:'eps')
                Output file type (Usually 'eps' or 'png')
        '''

        true_k = self.out['params']['true_k']
        opt_k = np.array(self.out['result']['avg_opt_k'])
        l_ddim = np.array(self.out['result']['avg_ddim'])

        ### Result of Diff
        t_criterion_diff = self.out['result']['evaluation']['t_criterion_diff']
        delay_diff = self.out['result']['evaluation']['delay_diff']
        benefit_diff = self.out['result']['evaluation']['benefit_diff']

        ### Result of TH
        t_criterion_th = self.out['result']['evaluation']['t_criterion_th']
        delay_th = self.out['result']['evaluation']['delay_th']
        benefit_th = self.out['result']['evaluation']['benefit_th']

        ### Result of weighted FS
        avg_ws_fs = self.out['result']['avg_ws_fs']

        ### Define directory name and file name
        now_time = dt.now()
        now_str = now_time.strftime("%Y%m%d-%H%M%S")
        date_str = now_time.strftime("%Y%m%d")

        dir_name = self.OUT_DIR
        if not os.path.isdir(dir_name):
            os.mkdir(dir_name)

        file_name = '%s/gmm_clu_se_ddim_%s'%(dir_name,type_name)
        out_fig = '%s_rand%d_alpha%.2f_%s.%s'%(file_name,self.random_state,self.alpha,now_str,file_format)

        ts = range(opt_k.shape[0])

        ### Define transition period
        trans_period = self._calc_trans_period(true_k=true_k)

        ts_sdms = []
        opt_k_sdms = []
        curr_o = None
        for i,(t,o) in enumerate(zip(ts,opt_k)):
            if i == 0:
                ts_sdms.append(t)
                opt_k_sdms.append(o)
                curr_o = o
                continue

            if curr_o == o:
                ts_sdms.append(t)
                opt_k_sdms.append(o)
                curr_o = o
            else:
                ts_sdms.append(t)
                ts_sdms.append(t)
                opt_k_sdms.append(curr_o)
                opt_k_sdms.append(o)
                curr_o = o
        ts_sdms = np.array(ts_sdms)
        opt_k_sdms = np.array(opt_k_sdms)

        alpha_true = 1.
        l_width = 5.
        ddim_width = 12.

        fig = plt.figure(figsize=(20,12))
        plt.rcParams["font.size"] = 24
        ax = fig.add_subplot(111)
        ax.plot(ts,true_k,'0.9',linewidth=20,alpha=alpha_true,label='True # of clusters')
        for tr in trans_period:
            plt.axvspan(tr[0], tr[1], color='0.9', alpha=alpha_true)

        str_ylabel = 'TH / Diff'
        title_str = 'Ddim for Gradual Change ( Transition Period: %s ) \n\
        <Symptom> TH:%s, Diff:%s, SDMS:%s' \
                            %(str(trans_period), str(self.out['result']['symptom_th']),
                            str(self.out['result']['symptom_diff']), str(self.out['result']['symptom_sdms']) )
        ax.plot(ts_sdms,opt_k_sdms,'b-',linewidth=l_width,label='SDMS')
        # ax.plot(ts,l_fs,'b--',linewidth=l_width,label='FS')
        ax.plot(ts,l_ddim,'g-', linewidth=ddim_width,label='Ddim')
        # ax.plot(ts,avg_ws_fs,'c:', linewidth=l_width,label='FS')
        plt.ylabel('SDMS / Ddim')
        plt.xlabel('Time')

        ax2 = ax.twinx()
        ax2.plot(ts,t_criterion_th,'r--', linewidth=l_width,label='TH')
        ax2.plot(ts,t_criterion_diff,'m-.', linewidth=l_width,label='Diff')
        cri_min, cri_max = 0, 1.5
        for c in self.out['result']['symptom_th']:
            ax2.plot([c,c],[cri_min,cri_max],'r--',linewidth=l_width)
        for c in self.out['result']['symptom_diff']:
            ax2.plot([c,c],[cri_min,cri_max],'m-.',linewidth=l_width)
        ax.legend(loc=2)
        ax2.legend(loc=4)
        plt.ylabel(str_ylabel)
        plt.title(title_str, fontsize=24)
        if tofile:
            plt.savefig(out_fig, format=file_format)
            plt.show()
        else:
            plt.show()


### Running this code on console
def define_argument():
    '''create definition of argument
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', help='conf file name')
    parser.add_argument('-p', '--pattern', default='single', help='change pattern \'s\' or \'m\'')
    parser.add_argument('-l', '--temp_beta', default=None, help='probability lambda')
    parser.add_argument('-n', '--num_data', default=1000, help='# of dataset')

    args = parser.parse_args()

    return parser, args


if __name__ == '__main__':
    parser, args = define_argument()

    sim_obj = simDdim(random_state=0,conf_file=args.file,c_pattern=args.pattern,
                      temp_beta=args.temp_beta,num_data=args.num_data,
                      sim_title='ddim',file_name='ddim_result')
    sim_obj.simulate()
