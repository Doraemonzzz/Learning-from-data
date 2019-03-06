# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 13:57:11 2019

@author: qinzhen
"""

import numpy as np
import matplotlib.pylab as plt
from numpy.linalg import inv
from cvxopt import matrix, solvers

#(c)
def generate(n, delta):
    X = np.random.uniform(size=n)
    epsilon = np.random.normal(size=n)
    y = X * X + delta * epsilon
    data = np.c_[X, y]
    return X, y, data

#参数
n = 50
delta = 0.1

#生成数据
X, y, data = generate(n, delta)

#构造D1,D2
a = np.array([0, 0.1])
D1 = data + a
D2 = data - a

plt.scatter(D1[:, 0], D1[:, 1], label='+a')
plt.scatter(D2[:, 0], D2[:, 1], label='-a')
plt.legend()
plt.show()

#(d)
#线性回归
#添加偏置项
X_treat = np.c_[np.ones(n), X]
w = inv(X_treat.T.dot(X_treat)).dot(X_treat.T).dot(y)

a1 = np.array([0, 1])
b1 = w[0] + w[1] * a1

plt.scatter(X, y)
plt.plot(a1, b1, 'r')
plt.show()

#使用此题介绍的分类方法
#Problem 3.5的算法2
def algorithm2(X, y):
    """
    算法2
    """
    N, d = X.shape
    A1 = X * y.reshape(-1, 1)
    A2 = np.c_[A1, (-1) * np.eye(N)]
    A3 = np.c_[np.zeros((N, d)), (-1) * np.eye(N)]
    
    A = np.r_[A2, A3]
    c = np.array([0.0] * d + [1.0] * N)
    b = np.array([-1.0] * N + [0.0]  * N)
    
    #带入算法求解
    c = matrix(c)
    A = matrix(A)
    b = matrix(b)
    
    sol = solvers.lp(c, A, b)
    
    #返回向量
    w = np.array((sol['x']))[:d]
    
    return w

#构造数据
X1 = np.r_[D1, D2]
X1 = np.c_[np.ones(2 * n), X1]
y1 = np.r_[np.ones(n), -1 * np.ones(n)]
    
#算法2
w2 = algorithm2(X1, y1)
#处理后的w
w_2 = - w2[:-1] / w2[-1]
a2 = np.array([0, 1])
b2 = w_2[0] + w_2[1] * a2
plt.scatter(X, y)
plt.plot(a2, b2, 'r')
plt.show()