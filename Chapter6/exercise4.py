# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 11:19:57 2018

@author: Administrator
"""

from scipy.special import comb, perm
import matplotlib.pyplot as plt
import numpy as np

def Q(k, x):
    s = 0
    for i in range(0, (k+1)//2):
        s1 = x**(i+1) * (1-x)**(k-i)
        s2 = (1-x)**(i+1) * x**(k-i)
        s += comb(k, i) * (s1 + s2)
    return s/x

x = np.arange(0.01, 0.5, 0.01)
K = np.arange(1, 20, 2)
for k in K:
    result = []
    for i in x:
        result.append(Q(k, i))
    plt.plot(x, result, label = str(k))
    plt.legend()
plt.show()

K = np.arange(1, 30, 2)
result = []
for k in K:
    a = perm(k, k)
    b = perm((k-1)//2, (k-1)//2)
    result.append(a / (b*b * 2**(k-1)))
plt.scatter(K, result)
plt.show()

def f(k, x):
    s = 0
    for i in range((k+1)//2, k+1):
        s1 = x**(i) * (1-x)**(k-i)
        s += comb(k, i) * s1
    return (1-2*x)*s/x

for k in K:
    result = []
    for i in x:
        result.append(f(k, i))
    plt.plot(x, result, label = str(k))
    plt.legend()
plt.show()