# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 12:27:30 2019

@author: qinzhen
"""

import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers
from helper import generatedata
from sklearn.preprocessing import PolynomialFeatures

def algorithm1(X, y):
    """
    算法1
    """
    N, d = X.shape
    c = np.array(np.ones(d))
    A = X * y.reshape(-1, 1)
    b = np.ones(N) * (-1.0)
    
    #转化为cvxopt中的数据结构
    c = matrix(c)
    A = matrix(A)
    b = matrix(b)
    
    sol = solvers.lp(c, A, b)
    
    w = np.array((sol['x']))
    
    return w

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

def draw(w, X, y, r, text, num):
    """
    作图
    """
    #作出直线
    a1 = np.array([-r, r])
    b1 = - (w[0] + w[1] * a1) / w[2]
    
    plt.scatter(X[y>0][:, 0], X[y>0][:, 1], s=1)
    plt.scatter(X[y<0][:, 0], X[y<0][:, 1], s=1)
    plt.plot(a1, b1, c="red")
    plt.title('sep={},algorithm{}'.format(text, num))
    plt.show()
    
    print(w)
    
# 定义等高线高度函数
def f(x1, x2, w):
    #将网格拉直并拼接
    X = np.c_[x1.reshape(-1, 1), x2.reshape(-1, 1)]
    #多项式转换
    poly = PolynomialFeatures(3)
    X_poly = poly.fit_transform(X)
    
    #计算结果
    result = X_poly.dot(w)
    #恢复成网格形状
    result = np.reshape(result, np.shape(x1))
    return result

#参数
rad = 10
thk = 5
sep = 5
N = 2000
r = 2 * (rad + thk)

# =============================================================================
# 特征转换之前
# =============================================================================
#产生数据
X, y = generatedata(rad, thk, sep, N)

#作图
plt.scatter(X[y>0][:, 0], X[y>0][:, 1], s=1)
plt.scatter(X[y<0][:, 0], X[y<0][:, 1], s=1)
plt.show()

#特征转换之前，算法1，sep=5
#对数据预处理，加上偏置项项1
X_treat = np.c_[np.ones(N), X]
w = algorithm1(X_treat, y)

#作图
draw(w, X, y, r, sep, 1)

#特征转换之前，算法2，sep=5
w = algorithm2(X_treat, y)

#作图
draw(w, X, y, r, sep, 2)

#特征转换之前，算法1，sep=-5
#产生数据
sep = -5
X, y = generatedata(rad, thk, sep, N)

#作图
plt.scatter(X[y>0][:, 0], X[y>0][:, 1], s=1)
plt.scatter(X[y<0][:, 0], X[y<0][:, 1], s=1)
plt.show()

#对数据预处理，加上偏置项项1
X_treat = np.c_[np.ones(N), X]
#特征转换之前，算法1，sep=-5
w = algorithm1(X_treat, y)
print(w)

#特征转换之前，算法2，sep=-5
w = algorithm2(X_treat, y)
draw(w, X, y, r, sep, 2)

# =============================================================================
# 特征转换后
# =============================================================================
#特征转换器
poly = PolynomialFeatures(3)

#特征转换后，算法1，sep=-5
#特征转换
sep = 5
X, y = generatedata(rad, thk, sep, N)
X_poly = poly.fit_transform(X)
w_poly = algorithm1(X_poly, y)

#数据数目
n = 2000

#定义a, b
a = np.linspace(-r, r, n)
b = np.linspace(-r, r, n)

#生成网格数据
A, B = np.meshgrid(a, b)

plt.contour(A, B, f(A, B, w_poly), 1, colors = 'red')
plt.scatter(X[y>0][:, 0], X[y>0][:, 1], s=1)
plt.scatter(X[y<0][:, 0], X[y<0][:, 1], s=1)
plt.title('featuretransform,sep=-5,algorithm1')
plt.show()
print(w_poly)

#特征转换后，算法2，sep=-5
#根据之前所述构造矩阵
w_poly = algorithm2(X_poly, y)

plt.contour(A, B, f(A, B, w_poly), 1, colors = 'red')
plt.scatter(X[y>0][:, 0], X[y>0][:, 1], s=1)
plt.scatter(X[y<0][:, 0], X[y<0][:, 1], s=1)
plt.title('featuretransform,sep=-5,algorithm2')
plt.show()
print(w_poly)