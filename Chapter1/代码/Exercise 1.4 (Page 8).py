# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 21:32:17 2019

@author: qinzhen
"""
import helper as hlp
import numpy as np
#%matplotlib inline
    
#设置随机种子，保证每次结果一致
seed = 42
rnd = np.random.RandomState(seed)
N = 20
d = 2
a, b, c, X, y, s, w = hlp.f(N, d, rnd)
hlp.plot_helper(a, b, c, X, y, s, w)