# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 14:35:47 2018

@author: Administrator
"""

import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

#参数
Qf = 9
N = 100
sigma2 = 1
D = 15
#np.random.seed(10)

#### Step 1:数据准备

#定义勒让德多项式，产生L(0,x),...,L(k,x),注意这里不要用递归
def L(k, x):
    if(k == 0):
        return [1.0]
    elif(k == 1):
        return [1.0, x*1.0]
    else:
        result = [1,x]
        for i in range(2, k+1):
            s = (2 * i - 1) / i * (x*result[-1]) - (i - 1) / i * result[-2]
            result.append(s)
        return result

#产生数据
def generate_data(Qf, N, sigma2, D):
    #系数ai
    a = np.random.normal(size=Qf+1)
    
    #标准化
    k = np.arange(1, 2*Qf+2, 2)
    s = (2 * a ** 2 / k).sum()
    a = a / np.sqrt(s)
    
    #产生点集
    x = np.random.uniform(low=-1,high=1,size=N)
    x.sort()
    #计算之前所述的X
    X = []
    for i in x:
        temp = L(Qf,i)
        X.append(temp)
    X = np.array(X)
    #差生误差项
    epsilon = np.sqrt(sigma2) * np.random.normal(size=N)
    #计算Y
    y = X.dot(a.T) + epsilon
    
    return x, y

#### Step 2:拟合数据

#对一个数据特征转换,将x转换为(1,x,...,x^k)
def t(x, k):
    result = [x**i for i in range(k+1)]
    return result

#对一组数据x=[x1,...xN]做特征转换
def tranform(X, k):
    result = []
    for x in X:
        temp = t(x,k)
        result.append(temp)
    return np.array(result)

#计算w=inv(X^T.X)X^Ty, H=Xinv(X^T.X+lambda.I)X^T
def H(X, y, Lambda=0):
    n = X.shape[1]
    w = inv(X.T.dot(X) + Lambda * np.eye(n)).dot(X.T).dot(y)
    H = X.dot(inv(X.T.dot(X) + Lambda * np.eye(n)).dot(X.T))
    return w, H

#利用d_eff, d_eff=trace(H^2)
def D_eff(H):
    return np.trace(H * H)

#计算p
def P(N, d_eff):
    return N / d_eff

#计算Ein
def Ein(X, y, Lambda=0):
    w, h = H(X, y, Lambda)
    N = X.shape[0]
    y1 = X.dot(w)
    Ein = np.mean((y1 - y) ** 2)
    d_eff = D_eff(h)
    p = P(N, d_eff)
    return Ein, p, w, d_eff

def VC_penality(Ein, p, d_eff, X):
    N = X.shape[0]
    return Ein * np.sqrt(p) / (np.sqrt(p) - np.sqrt(1 + np.log(p) + np.log(N) / (2 * d_eff)))

def FPE(Ein, p):
    return (p + 1) / (p - 1) * Ein

def LOO_CV(X, y, Lambda=0):
    N = X.shape[0]
    E = []
    for i in range(N):
        index = np.array([True] * N)
        index[i] = False
        X_test = X[i]
        y_test = y[i]
        X_train = X[index]
        y_train = y[index]
        w, h = H(X_train, y_train, Lambda)
        y_pred = X_test.dot(w)
        E.append((y_pred - y_test) ** 2)
    return np.mean(E)

def Permutation_estimate(X, y, Ein, w):
    N = X.shape[0]
    index = np.arange(0, N)
    permutation = np.random.permutation(index)
    X1 = X[permutation]
    y1 = y[permutation]
    
    y_mean = np.mean(y)
    permutation_penalty = np.sum((y1 - y_mean) * (X1.dot(w)))
    return Ein + permutation_penalty / (2 * N)

def simulation_1(Qf, N, sigma2, D):
    #产生数据
    x, y = generate_data(Qf, N, sigma2, D)
    
    #生成变换矩阵
    degree = np.arange(1, D + 1)
    X_trans = []
    for i in degree:
        X_trans.append(tranform(x, i))
    
    #模拟实验
    vc = []
    fpe = []
    loo_cv = []
    permutation = []
    for i in range(D):
        E_in, p, w, d_eff =  Ein(X_trans[i], y)
        #VC
        vc_penality = VC_penality(E_in, p, d_eff, X_trans[i])
        vc.append(vc_penality)
        
        #FPE
        fpe_penality = FPE(E_in, p)
        fpe.append(fpe_penality)
        
        #LOO-CV
        cv = LOO_CV(X_trans[i], y)
        loo_cv.append(cv)
        
        #Permutation_estimate
        per = Permutation_estimate(X_trans[i], y, E_in, w)
        permutation.append(per)
    return vc, fpe, loo_cv, permutation

#(a)
'''
vc = []
fpe = []
loo_cv = []
permutation = []
    
for i in range(50):
    result = simulation_1(Qf, N, sigma2, D)
    vc.append(result[0])
    fpe.append(result[1])
    loo_cv.append(result[2])
    permutation.append(result[3])

vc = np.array(vc)
fpe = np.array(fpe)
loo_cv = np.array(loo_cv)
permutation = np.array(permutation)

vc = np.mean(vc, axis=0)
fpe = np.mean(fpe, axis=0)
loo_cv = np.mean(loo_cv, axis=0)
permutation = np.mean(permutation, axis=0)

degree = np.arange(1, D + 1)
plt.plot(degree, vc, label="$E_{vc}$")
plt.plot(degree, fpe, label="$E_{fpe}$")
plt.plot(degree, loo_cv, label="$E_{cv}$")
plt.plot(degree, permutation, label="$E_{perm}$")
plt.legend()
plt.show()
'''

#(b)
def simulation_2(Qf, N, sigma2, D, Lambda):
    #产生数据
    x, y = generate_data(Qf, N, sigma2, D)
    
    #生成变换矩阵
    X_trans = tranform(x, D)
    
    #系数范围
    n = Lambda.shape[0]
    
    #模拟实验
    vc = []
    fpe = []
    loo_cv = []
    permutation = []
    for i in range(n):
        E_in, p, w, d_eff =  Ein(X_trans, y, Lambda[i])
        #VC
        vc_penality = VC_penality(E_in, p, d_eff, X_trans)
        vc.append(vc_penality)
        
        #FPE
        fpe_penality = FPE(E_in, p)
        fpe.append(fpe_penality)
        
        #LOO-CV
        cv = LOO_CV(X_trans, y, Lambda[i])
        loo_cv.append(cv)
        
        #Permutation_estimate
        per = Permutation_estimate(X_trans, y, E_in, w)
        permutation.append(per)
    return vc, fpe, loo_cv, permutation

vc = []
fpe = []
loo_cv = []
permutation = []
D = 8
Qf = 5
N = 15
Lambda = np.linspace(0, 5, 15)
    
for i in range(500):
    result = simulation_2(Qf, N, sigma2, D, Lambda)
    vc.append(result[0])
    fpe.append(result[1])
    loo_cv.append(result[2])
    permutation.append(result[3])

vc = np.array(vc)
fpe = np.array(fpe)
loo_cv = np.array(loo_cv)
permutation = np.array(permutation)

vc = np.mean(vc, axis=0)
fpe = np.mean(fpe, axis=0)
loo_cv = np.mean(loo_cv, axis=0)
permutation = np.mean(permutation, axis=0)


plt.plot(Lambda, vc, label="$E_{vc}$")
plt.plot(Lambda, fpe, label="$E_{fpe}$")
plt.plot(Lambda, loo_cv, label="$E_{cv}$")
plt.plot(Lambda, permutation, label="$E_{perm}$")
plt.legend()
plt.show()