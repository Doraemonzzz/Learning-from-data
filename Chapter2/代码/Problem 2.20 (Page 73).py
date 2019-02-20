# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 15:19:48 2019

@author: qinzhen
"""

from scipy.special import comb
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

dvc = 50
delta = 0.05

#计算m(N)
def m(n):
    k = min(dvc, n)
    s = 0
    for i in range(k+1):
        s += comb(n, k)
    return s

#Original VC-bound
def f1(n):
    result = (8 / n) * np.log(4 * m(2 * n) / delta)
    result = result ** 0.5
    return result

#Rademacher Penalty Bound
def f2(n):
    k1 = (2 * np.log(2 * n * m(n)) / n)
    k2 = (2 / n) * np.log(1 / delta)
    k3 = 1 / n
    result = k1 ** 0.5 + k2 ** 0.5 + k3
    return result

#Parrondo and Van den Broek
def f3(n):
    k1 = 1 / n
    k2 = 1 / (n ** 2) + (1 / n) * np.log(6 * m(2 * n) / delta)
    k2 = k2 ** 0.5
    result = k1 + k2
    return result

#Devroye
def f4(n):
    k1 = 1 / (n - 2)
    k2 = np.log(4 * m(n ** 2) / delta) / (2 * (n - 2)) + 1 / ((n - 2) ** 2)
    k2 = k2 ** 0.5
    result = k1 + k2
    return result

#产生点集
x = np.arange(100, 2000)

y1 = [f1(i) for i in x]
y2 = [f2(i) for i in x]
y3 = [f3(i) for i in x]
y4 = [f4(i) for i in x]

plt.plot(x, y1, label="Original VC-bound")
plt.plot(x, y2, label="Rademacher Penalty Bound")
plt.plot(x, y3, label="Parrondo and Van den Broek")
plt.plot(x, y4, label="Devroye")
plt.legend()
plt.show()

#比较y3, y4
plt.plot(x, y3, label="Parrondo and Van den Broek")
plt.plot(x, y4, label="Devroye")
plt.legend()
plt.show()