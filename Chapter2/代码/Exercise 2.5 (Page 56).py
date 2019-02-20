# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 21:46:18 2019

@author: qinzhen
"""

import numpy as np
def m(n):
    return n + 1

def delta(N, epsilon):
    return 4 * m(2 * N) / (np.exp(N * (epsilon ** 2) / 8))

print(delta(100, 0.1))