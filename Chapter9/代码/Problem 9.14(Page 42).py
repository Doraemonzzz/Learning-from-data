# -*- coding: utf-8 -*-
"""
Created on Thu May  2 23:00:48 2019

@author: qinzhen
"""

import numpy as np
import matplotlib.pyplot as plt

#(a)
def f(N):
    #生成数据
#    N = 40
    X = np.random.randn(N)
    r = np.sign(np.random.rand(N) - 0.5)
    
    #寻找阈值
    X1 = np.sort(X)
    X2 = (X1[1:] + X1[:-1]) / 2
    X2 = np.append(X1[0] - 1, X2)
    X2 = np.append(X2, X1[-1] + 1)
    
    #计算结果，向量化计算
    temp = np.sign(X1.reshape(-1, 1) - X2)
    result = np.mean(temp != r.reshape(-1, 1), axis=0)
    error = np.min(result)
    
    return 1 / 2 - error

N = np.arange(1, 101)
Error = []
for n in N:
    Error.append(f(n))
plt.plot(N, Error)
plt.title("penalty VS N")
plt.show()

#(b)
m = 100
Error = []
for n in N:
    error = []
    for _ in range(m):
        error.append(f(n))
    Error.append(np.mean(error))
plt.plot(N, Error)
plt.title("mean penalty VS N")
plt.show()

#(c)
N1 = 1 / np.sqrt(N)
N2 = np.sqrt(8 * np.log((4 * ((2 * N) ** 2 + 1))) / N)
plt.plot(N, Error, label="Rademacher optimism penalty")
plt.plot(N, N1, label="$1/\sqrt{N}$")
plt.plot(N, N2, label="VC penalty")
plt.title("mean penalty VS N")
plt.legend()
plt.show()