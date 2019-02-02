
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

class evalSymptom():
    '''Evaluation of symptom detection
    '''

    def __init__(self):
        '''constructor'''

    def symptom_detection(self,t_criterion,th_change=.1,):
        '''Detect symptom
        [Parameters]
            t_criterion: list, shape (time)
                The value of criterion in each time
            th_change: float, optional (default:0.1)
                The threshold to detect symptom
        '''

        symptom = []
        continue_change = False
        for t,cri in enumerate(t_criterion):

            if not continue_change and cri >= th_change:
                continue_change = True
                symptom.append(t)
                continue

            if continue_change and cri >= th_change:
                continue

            if cri < th_change:
                continue_change = False

        return symptom

    def eval_symptom(self,grad_range,symptom,ben_T=10.):
        '''Evaluate symptom
        [Parameters]
            grad_range: list, shape ( # of transition period, 2)
                List of transition period
            symptom: list
                The result of detecting symptom
            ben_T: float, optional (default:10.0)
                The range of calculating benefit
        '''

        delay_value = [np.nan] * len(grad_range)
        benefit_value = [np.nan] * len(grad_range)
        for c in symptom:
            bool_far = True
            for i,g in enumerate(grad_range):
                if c in range(g[0],g[1]+1):
                    bool_far = False
                    if np.isnan(delay_value[i]):
                        delay_value[i] = np.float(c-g[0])
                        benefit_value[i] = np.max([0.,1.-np.float(c-g[0])/ben_T])

        return delay_value, benefit_value
