# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 23:08:39 2019

@author: qinzhen
"""

from scipy.special import comb
import matplotlib.pyplot as plt

def m(N, d):
    result = 0
    k = min(d, N-1)
    for i in range(k + 1):
        result += comb(N-1, i)
    return 2 * result

x = range(1,41)
d = 10
y = [m(i, d) / (2 ** i) for i in x]

plt.plot(x,y)
plt.xlabel('N')
plt.ylabel('mH(N)/2**N')
plt.show()

print([m(i, d) / (2 ** i) for i in [10, 20, 40]])