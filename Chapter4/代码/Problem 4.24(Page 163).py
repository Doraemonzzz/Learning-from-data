# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 16:29:00 2019

@author: qinzhen
"""

import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

d = 3
N = range(d+15, d+116, 10)

#计算一次实验结果
def process(n, d=3, sigma=0.5, k=0.05):
    #生成点
    X = np.random.normal(size=(n, d))
    #偏置项
    bias = np.ones(n)
    #合并X和bias
    X = np.c_[bias, X]
    #正则项系数
    k = k / n
    #权重
    w = np.zeros(d+1)
    #生成噪声
    epsilon = np.random.normal(size=n)
    #生成y
    y = X.dot(w) + sigma * epsilon
    e1 = 0
    e2 = 0
    E = np.array([])
    #交叉验证
    for i in range(n):
        X1 = np.r_[X[:i, :], X[i+1:, :]]
        y1 = np.r_[y[:i], y[i+1:]]
        w1 = inv(X1.T.dot(X1) + k * np.eye(d+1)).dot(X1.T.dot(y1))
        e = (X[i].dot(w1) - y[i]) ** 2
        E = np.append(E, e)
        if(i == 1):
            e1 = e
        elif i == 2:
            e2 = e
    return e1, e2, np.mean(E)

#(a)
#记录每个N对应的e1,e2,ecv
E = {}
for n in N:
    E[n] = []

for n in N:
    E1 = np.array([])
    E2 = np.array([])
    Ecv = np.array([])
    for i in range(3000):
        e1, e2, ecv = process(n)
        E1 = np.append(E1, e1)
        E2 = np.append(E2, e2)
        Ecv = np.append(Ecv, ecv)
    #计算均值和方差
    mean = (E1.mean(), E2.mean(), Ecv.mean())
    var = (E1.var(), E2.var(), Ecv.var())
    E[n].append(mean)
    E[n].append(var)

#(b)
meanE1 = np.array([])
meanE2 = np.array([])
meanEcv = np.array([])
for n in N:
    mean = E[n][0]
    meanE1 = np.append(meanE1, mean[0])
    meanE2 = np.append(meanE2, mean[1])
    meanEcv = np.append(meanEcv,mean[2])
plt.plot(N, meanE1, label='e1')
plt.plot(N, meanE2, label='e2')
plt.plot(N, meanEcv, label='ecv')
plt.title('mean')
plt.xlabel('N')
plt.legend()
plt.show()

#(e)
varE1 = np.array([])
varE2 = np.array([])
varEcv = np.array([])
for n in N:
    var = E[n][1]
    varE1 = np.append(varE1, var[0])
    varE2 = np.append(varE2, var[1])
    varEcv = np.append(varEcv, var[2])
plt.plot(N, N, label='$N$')
plt.plot(N, varE1 / varEcv, label='$N_{eff}$')
plt.legend()
plt.show()  