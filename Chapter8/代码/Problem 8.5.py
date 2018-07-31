# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 11:36:01 2018

@author: Administrator
"""

import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt

####a,b
def generate(n = 3):
    """生成n个点"""
    X1 = np.array([])
    Y1 = np.array([])
    X2 = np.array([])
    Y2 = np.array([])
    for i in range(n):
        x1 = np.random.uniform(-1,1)
        y1 = np.random.uniform(0,1)
        x2 = np.random.uniform(-1,1)
        y2 = np.random.uniform(-1,0)
        X1 = np.append(X1,x1)
        X2 = np.append(X2,x2)
        Y1 = np.append(Y1,y1)
        Y2 = np.append(Y2,y2)
    return X1,Y1,X2,Y2

#生成数据
X1,Y1,X2,Y2 = generate()
plt.scatter(X1,Y1)
plt.scatter(X2,Y2)
plt.show()

#产生结果
def mysvm(Y1,Y2):
    """找到+1类中纵坐标最小的点，-1类中纵坐标最大的点"""
    y1 = Y1.copy()
    y2 = Y2.copy()
    y1.sort()
    y2.sort()
    return (y1[0] + y2[-1])/2

a_random = np.random.uniform(-1,1)
a_svm = mysvm(Y1,Y2)

plt.scatter(X1,Y1,label = "+")
plt.scatter(X2,Y2,label = "-")
plt.plot([-1,1],[a_random,a_random],label = "random")
plt.plot([-1,1],[a_svm,a_svm],label = "svm")
plt.legend()
plt.show()

print("a_random =",a_random)
print("a_svm =",a_svm)

#c,d
N = 100000
A_random = []
A_svm = []
for i in range(N):
    X1,Y1,X2,Y2 = generate()
    a_random = np.random.uniform(-1,1)
    a_svm = mysvm(Y1,Y2)
    A_random.append(a_random)
    A_svm.append(a_svm)

#画直方图
plt.hist(A_random,label = 'random')
plt.hist(A_svm,label = 'svm')
plt.legend()
plt.show()

#e
#计算a_random_mean,a_svm_mean
A_random = np.array(A_random)
A_svm = np.array(A_svm)
a_random_mean = np.mean(A_random)
a_svm_mean = np.mean(A_svm)

X_2 = np.random.uniform(-1,1,100000)
Y = np.sign(X_2)
Y_random_mean = np.sign(X_2 - a_random_mean)
Y_svm_mean = np.sign(X_2 - a_svm_mean)

bias_random = np.mean(Y != Y_random_mean)
bias_svm = np.mean(Y != Y_svm_mean)
print("bias_random =",bias_random)
print("bias_svm =",bias_svm)

#计算var_random_mean,var_svm
var_random = np.array([])
var_svm = np.array([])
for i in range(1000):
    X1,Y1,X2,Y2 = generate()
    a_random = np.random.uniform(-1,1)
    a_svm = mysvm(Y1,Y2)
    X = np.random.uniform(-1,1,1000)
    Y_random = np.sign(X - a_random)
    Y_svm = np.sign(X - a_svm)
    Y_random_mean = np.sign(X - a_random_mean)
    Y_svm_mean = np.sign(X - a_svm_mean)
    var_random = np.append(var_random,np.mean(Y_random_mean != Y_random))
    var_svm = np.append(var_svm,np.mean(Y_svm_mean != Y_svm))
    
print("var_svm =",np.mean(var_svm))
print("var_random =",np.mean(var_random))