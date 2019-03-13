# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 17:35:52 2019

@author: qinzhen
"""

import numpy as np
from numpy.linalg import inv
from scipy.integrate import quad
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

#参数
Qf = 5
N = 100
sigma2 = 1

#### Step 1:数据准备

#定义勒让德多项式，产生L(0,x),...,L(k,x),注意这里不要用递归
def L(k,x):
    if(k == 0):
        return [1.0]
    elif(k == 1):
        return [1.0, x * 1.0]
    else:
        result = [1, x]
        for i in range(2,k+1):
            s = (2 * i - 1) / i * (x * result[-1]) - (i - 1) / i * result[-2]
            result.append(s)
        return result

#系数ai
a = np.random.normal(size=Qf+1)

#标准化
k = np.arange(1, 2*Qf+2, 2)
s = (2 * a**2 / k).sum()
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
Y1 = X.dot(a.T)
Y = Y1 + epsilon


#### Step 2:拟合数据

#对一个数据特征转换,将x转换为(1,x,...,x^k)
poly2 = PolynomialFeatures(2)
poly10 = PolynomialFeatures(10)

#特征转换
X2 = poly2.fit_transform(x)
X10 = poly10.fit_transform(x)

#计算结果
w2 = inv(X2.T.dot(X2)).dot(X2.T).dot(Y)
w10 = inv(X10.T.dot(X10)).dot(X10.T).dot(Y)


#### Step 3:计算结果

#构造被积函数,a为系数
def E(x, w, a, sigma2):
    """
    被积函数
    """
    #计算f(x)
    #n为勒让德多项式次数
    n = len(a)-1
    l = L(n, x)
    f = a.dot(l)
    
    #计算g(x)
    X = np.array([x**i for i in range(len(w))])
    g = X.dot(w)
    
    return (g - f) ** 2 / 2 + sigma2

E2 = quad(E, -1, 1, args=(w2, a, sigma2))[0]
E10 = quad(E, -1, 1, args=(w10, a, sigma2))[0]

print("E2=" + str(E2))
print("E10=" + str(E10))
print("E10-E2=" + str(E10 - E2))

#### Step 4:作图
Y2 = X2.dot(w2)
Y10 = X10.dot(w10)

plt.plot(x, Y1, label='无噪音')
plt.scatter(x, Y, label='有噪音',s=3)
plt.plot(x, Y2, label='g2')
plt.plot(x, Y10, label='g10')
plt.legend()
plt.show()