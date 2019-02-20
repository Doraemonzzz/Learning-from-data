# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 11:03:25 2019

@author: qinzhen
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

def Pmax(N, x, u):
    low = int(3 - 6*x)
    up = int(3 + 6*x)
    s = 0
    for k in range(low, up+1):
        s += comb(N, k) * (u ** k) * ((1 - u) ** (N - k))
    return 1 - s ** 2

x = np.arange(0, 1, 0.01)
y1 = np.array([Pmax(6, i, 0.5) for i in x])
y2 = np.array([2 * np.exp(-2 * 6 * (i ** 2)) for i in x])

plt.plot(x, y1, label="Pmax")
plt.plot(x, y2, label="Hoeffding")
plt.xlabel("eplison")
plt.ylabel("P")
plt.legend()
plt.show()