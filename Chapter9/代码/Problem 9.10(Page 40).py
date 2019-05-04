# -*- coding: utf-8 -*-
"""
Created on Sat May  4 09:12:52 2019

@author: qinzhen
"""

#### (b)
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
import helper as hlp

#### (i)
N = 40
d = 5
k = 3

def data(N, d):
    """
    生成数据集
    """
    X = np.random.randn(N, d)
    w = np.random.randn(d)
    epsilon = np.random.randn(N) * 0.5
    y = X.dot(w) + epsilon
    return w, X, y

#### (ii)
#Algorithm 1
def Algorithm_1(X, y, k):
    pca = hlp.PCA_(n_components=k)
    pca.fit(X)
    Z = pca.fit_transform(X)
    e1 = []
    N, d = X.shape
    for i in range(N):
        #每轮选择的数据下标
        index = np.array([True] * N)
        index[i] = False
        #划分数据
        Z0 = Z[i]
        y0 = y[i]
        Z1 = Z[index]
        y1 = y[index]
        w = inv(Z1.T.dot(Z1)).dot(Z1.T).dot(y1)
        e1.append((Z0.dot(w) - y0) ** 2)
    return np.mean(e1)

#Algorithm 2
def Algorithm_2(X, y, k):  
    e2 = []
    
    N, d = X.shape
    for i in range(N):
         #每轮选择的数据下标
        index = np.array([True] * N)
        index[i] = False
        #划分数据
        X0 = X[i].reshape(1, -1)
        y0 = y[i]
        X1 = X[index]
        y1 = y[index]
        #训练
        pca = hlp.PCA_(n_components=k)
        pca.fit(X1)
        Z1 = pca.transform(X1)
        w = inv(Z1.T.dot(Z1)).dot(Z1.T).dot(y1)
        
        Z0 = pca.transform(X0)
    
        e2.append((Z0.dot(w) - y0) ** 2)
    return np.mean(e2)

#### (iii)
def E_out(X, y, w):
    #计算结果
    w0 = inv(X.T.dot(X)).dot(X.T).dot(y)
    #生成新的数据来模拟Eout
    d = X.shape[1]
    N = 10000
    X1 = np.random.randn(N, d)
    epsilon = np.random.randn(N) * 0.5
    y1 = X1.dot(w) + epsilon
    y0 = X1.dot(w0)
    return np.mean((y1 - y0) ** 2)

#### (iv)
w, X, y = data(N, d)
print("E1 =", Algorithm_1(X, y, k))
print("E2 =", Algorithm_2(X, y, k))
print("E_out =", E_out(X, y, w))


#### (v)
M = 1000
E1 = []
E2 = []
Eout = []
for i in range(M):
    w, X, y = data(N, d)
    E1.append(Algorithm_1(X, y, k))
    E2.append(Algorithm_2(X, y, k))
    Eout.append(E_out(X, y, w))
    
plt.hist(E1)
plt.title("E1")
plt.show()
plt.hist(E2)
plt.title("E2")
plt.show()
plt.hist(Eout)
plt.title("Eout")
plt.show()
print("E1_mean =", np.mean(E1))
print("E2_mean =", np.mean(E2))
print("E_out_mean =", np.mean(Eout))