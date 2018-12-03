# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 13:29:00 2018

@author: Administrator
"""

import matplotlib.pyplot as plt
from scipy.special import comb

def f(d):
    s1 = 8**d
    s2 = 0
    for l in range(d+1):
        for m in range(d+1):
            for k in range(d+1):
                if(l > m + k + 1):
                    s2 += comb(d, l) * comb(d, m) * comb(d, k)
    return 1 - s2 / s1

D = range(10, 61, 5)
P = [f(d) for d in D]

plt.plot(D, P)