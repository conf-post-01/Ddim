# -*- coding: utf-8 -*-

import sys
import os
import numpy as np
import pandas as pd
import math
import time
import json
from datetime import datetime as dt
import configparser


class calcGauNormNML:
    '''Calculate normalization term of Gaussian NML
    '''

    def __init__(self):
        '''constructor
        '''
        self.BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.DATA_DIR = os.path.normpath(os.path.join(self.BASE_DIR, './data'))
        self.OUT_DIR = os.path.normpath(os.path.join(self.DATA_DIR, './out'))
        self.CONF_DIR = os.path.normpath(os.path.join(self.BASE_DIR, './conf'))

        conf_file = configparser.ConfigParser()
        conf_file.read(self.CONF_DIR+'/calcGaussianNML.conf')

        self.NUM_DATA = int(conf_file.get('data_param', 'NUM_DATA'))
        self.NUM_CLUSTER = int(conf_file.get('data_param', 'NUM_CLUSTER'))

        ### Gaussian Mixture Model parameters
        self.DIM = int(conf_file.get('gaussian_param', 'DIM'))
        self.PARA_R = np.float(conf_file.get('gaussian_param', 'PARA_R'))
        self.PARA_LAMBDA = np.float(conf_file.get('gaussian_param', 'PARA_LAMBDA'))

        self.start = time.time()

    def calc_fun_log_I(self,h):
        '''Calculate function log(I) with log scale
        '''
        if h >= self.DIM:
            m = float(self.DIM)
            cont = (m+1)*np.log(2) + m*np.log(self.PARA_R)/2. \
                    - m**2 * np.log(self.PARA_LAMBDA) / 2. - (m+1)*np.log(m) - math.lgamma(m/2.) \
                    + m*h*np.log(h/2/math.e)/2.
            l_log_gamma = [math.lgamma((h-i+1)/2.) for i in range(1,self.DIM+1)]
            log_gamma_m = m*(m-1)/4.*np.log(math.pi) + np.sum(l_log_gamma)

            value_I = cont - log_gamma_m
        else:
            value_I = 0.0

        return value_I

    def calc_log_n_choose_r(self,n_num):
        '''Calculate the combination with log scale
        [Paramters]
            n_num: int
                The number of dataset
        '''
        if n_num == 0:
            l_sum_ncr = [0.]
        else:
            tmp_range = np.log( range(1,n_num+1) ).tolist()
            l_range = [0]
            l_range.extend(tmp_range)
            l_range = np.array(l_range)

            l_all = sum(l_range) * np.ones(n_num+1)
            l_denominator = l_range.dot(np.tri(n_num+1).T)
            l_sum_ncr = l_all - l_denominator - l_denominator[::-1]

        return l_sum_ncr

    def calc_log_list(self,n_num):
        '''Calculate the formula: r*log(r/n)+(n-r)*log((n-r)/n)
        [Paramters]
            n_num: int
                The number of dataset
        '''
        if n_num == 0:
            l_all_log = [0.]
        else:
            tmp_range = np.log( range(1,n_num+1) ).tolist()
            l_range = [0]
            l_range.extend(tmp_range)
            l_range = np.array(l_range)

            l_log = np.array(range(n_num+1) * l_range)
            l_all_log = l_log + l_log[::-1] - n_num * np.log(n_num)

        return l_all_log

    def calc_norm_NML(self):
        '''Calculate normalization term
        '''
        fun_I = np.vectorize(lambda x: self.calc_fun_log_I(x))

        df_norm_NML = pd.DataFrame()
        df_norm_NML["cluster_1"] = pd.Series(fun_I(range(self.NUM_DATA+1)))

        for clu in range(2,self.NUM_CLUSTER+1):
            bef_cluster = "cluster_" + str(clu-1)
            now_cluster = "cluster_" + str(clu)

            df_L_list = pd.DataFrame(np.zeros((self.NUM_DATA+1,self.NUM_DATA+1)))
            for n_num in range(self.NUM_DATA+1):
                l_sum_ncr = self.calc_log_n_choose_r(n_num)
                l_all_log = self.calc_log_list(n_num)
                l_log_L = np.array(l_sum_ncr) + np.array(l_all_log) \
                            + np.array(df_norm_NML.ix[range(n_num+1),bef_cluster]) \
                            + np.array(fun_I(range(n_num,-1,-1)))
                df_L_list.ix[n_num,range(n_num+1)] = l_log_L
                df_L_list.ix[n_num,range(n_num+1,df_L_list.shape[0])] = -np.inf

            df_norm_NML[now_cluster] = pd.Series(df_L_list.apply(lambda x: self.log_sum(x),axis=1))

        self.elapsed_time = time.time() - self.start

        return df_norm_NML

    def log_sum(self,l_log):
        '''Calculate sum of values with log scale
        [Paramters]
            l_log: list
                Values with log scale
        '''
        max_log = max(l_log)
        l_log_minus = l_log - max_log
        out_value = max_log + np.log(np.sum(np.exp(l_log_minus)))

        return out_value

    def out_files(self,df_norm_NML):
        '''output files
        [Paramters]
            df_norm_NML: pd.DataFrame, shape ( # of dataset + 1, # of clusters)
                Calculated normalization term
        '''
        now_time = dt.now()
        now_str = now_time.strftime("%Y%m%d-%H%M%S")
        out_dict = {'param':{'dim':self.DIM,
                             'num_cluster':self.NUM_CLUSTER,
                             'para_R':self.PARA_R,
                             'para_lambda':self.PARA_LAMBDA,
                             'num_data':self.NUM_DATA,
                             'time':self.elapsed_time
                            },
                    'normNML':{}}
        out_name = 'normNML_Gaussian'
        file_name = self.OUT_DIR + '/' + out_name + '_' + \
                    'N' + str(self.NUM_DATA) + 'd' + str(self.DIM) + 'K' + str(self.NUM_CLUSTER) + '_' + now_str + '.json'

        for clu in df_norm_NML.columns:
            out_dict['normNML'][clu] = list(df_norm_NML[clu])

        with open(file_name, 'w') as f:
            json.dump(out_dict, f, sort_keys=True, indent=4)

def main():
    '''main runner
    '''
    nml_obj = calcGauNormNML()
    df_norm_NML = nml_obj.calc_norm_NML()
    nml_obj.out_files(df_norm_NML)

if __name__ == '__main__':
    main()
