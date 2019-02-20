# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 23:14:03 2019

@author: qinzhen
"""

import numpy as np

delta = 0.1

def f(N, delta):
    return np.sqrt(8 / N * np.log(4 * (2 * N + 1) /delta))

print(f(100, delta))

print(f(10000, delta))