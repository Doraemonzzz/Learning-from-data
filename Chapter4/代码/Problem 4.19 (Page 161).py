# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 19:49:28 2019

@author: qinzhen
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from helper import L, E


def process(Lambda, Qf=5, N=100, sigma2=1, d=2):
    """
    进行一次实验，Lambda为正则项的系数，Qf为勒让德多项式的次数，
    N为数据数量，sigma2为方差，d为特征转换的次数
    """
    #### Step 1:数据准备
    #系数ai
    a = np.random.normal(size=Qf+1)
    
    #标准化
    k = np.arange(1, 2*Qf+2, 2)
    s = (2*a**2/k).sum()
    a = a / np.sqrt(s)
    
    #产生点集
    x = np.random.uniform(low=-1, high=1, size=N)
    x.sort()
    x = x.reshape(-1, 1)
    #计算之前所述的X
    X = []
    for i in x:
        temp = L(Qf, i)
        X.append(temp)
    X = np.array(X)
    #差生误差项
    epsilon = np.sqrt(sigma2) * np.random.normal(size=N)
    #计算Y
    Y = X.dot(a.T) + epsilon
    
    #### Step 2:拟合数据
    polyd = PolynomialFeatures(d)
    
    #特征转换
    Xd = polyd.fit_transform(x, d)
    
    #计算Lasso回归,Ridge回归
    #Lasso
    lasso=linear_model.Lasso(alpha=Lambda)
    lasso.fit(Xd, Y)
    
    #Ridge
    ridge=linear_model.Ridge(alpha=Lambda)
    ridge.fit(Xd, Y)
    
    #### Step 3:计算结果
    
    El=quad(E, -1, 1, args=(lasso.coef_, a, sigma2))[0]
    Eq=quad(E, -1, 1, args=(ridge.coef_, a, sigma2))[0]

    return El, Eq, lasso.coef_, ridge.coef_

#(a)
Lambda = np.arange(0.01,2.01,0.05)
El = []
Eq = []
lasso = []
ridge = []
for l in Lambda:
    result = process(l)
    El.append(result[0])
    Eq.append(result[1])
    
plt.plot(Lambda, El, label='Lasso')
plt.plot(Lambda, Eq, label='Ridge')
plt.xlabel('$\lambda$')
plt.title('$E_{out}$')
plt.legend()
plt.show()

#(c)
lasso = []
ridge = []
for l in Lambda:
    result = process(l, Qf=20, N=3, d=5)
    lasso.append(np.sum(result[2] != 0))
    ridge.append(np.sum(result[3] != 0))
plt.plot(Lambda, lasso, label='lasso')
plt.plot(Lambda, ridge, label='ridge')
plt.title('number of non-zero weights')
plt.legend()
plt.xlabel('$\lambda$')
plt.show()