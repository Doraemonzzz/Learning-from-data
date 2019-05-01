# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 15:45:18 2019

@author: qinzhen
"""

from scipy.special import comb
import matplotlib.pyplot as plt
import numpy as np

#(b)
#计算Qk
def Q(k, x):
    s = 0
    for i in range(0, (k+1)//2):
        s1 = x ** (i + 1) * (1 - x) ** (k - i)
        s2 = (1 - x) ** (i + 1) * x ** (k - i)
        s += comb(k, i) * (s1 + s2)
    return s

x = np.arange(0.01, 0.5, 0.01)
K = np.arange(1, 20, 2)
for k in K:
    result = []
    for i in x:
        result.append(Q(k, i))
    plt.plot(x, result, label = str(k))
    plt.legend()
plt.title("$Q_k$ VS $\eta$")
plt.show()