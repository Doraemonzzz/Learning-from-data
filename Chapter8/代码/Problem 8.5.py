# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 12:38:01 2019

@author: qinzhen
"""

import numpy as np
import matplotlib.pyplot as plt

####a,b
def generate(n=3):
    """
    生成n个点
    """
    X1p = np.random.uniform(-1, 1, n)
    X2p = np.random.uniform(0, 1, n)
    X1n = np.random.uniform(-1, 1, n)
    X2n = np.random.uniform(-1, 0, n)

    return X1p, X2p, X1n, X2n

#生成数据
X1p, X2p, X1n, X2n = generate()
plt.scatter(X1p, X2p)
plt.scatter(X1n, X2n)
plt.show()

#产生结果
def mysvm(X2p, X2n):
    """找到+1类中纵坐标最小的点，-1类中纵坐标最大的点"""
    return (np.min(X2p) + np.max(X2n)) / 2

a_random = np.random.uniform(-1, 1)
a_svm = mysvm(X2p, X2n)

plt.scatter(X1p, X2p, label="+")
plt.scatter(X1n, X2n, label="-")
plt.plot([-1, 1], [a_random, a_random], label="random")
plt.plot([-1, 1], [a_svm, a_svm], label="svm")
plt.legend()
plt.show()

print("a_random = {}".format(a_random))
print("a_svm = {}".format(a_svm))

####c,d
N = 100000
A_random = []
A_svm = []
for i in range(N):
    X1p, X2p, X1n, X2n = generate()
    a_random = np.random.uniform(-1, 1)
    a_svm = mysvm(X2p, X2n)
    A_random.append(a_random)
    A_svm.append(a_svm)

#画直方图
plt.hist(A_random, label='random')
plt.hist(A_svm, label='svm')
plt.legend()
plt.show()

####e
#计算a_random_mean,a_svm_mean
#根据之前模拟的结果得到随机选择以及svm算法对应的系数
A_random = np.array(A_random)
A_svm = np.array(A_svm)
a_random_mean = np.mean(A_random)
a_svm_mean = np.mean(A_svm)
#生成用于模拟的数据
X2 = np.random.uniform(-1, 1, 100000)
#计算标签
Y = np.sign(X2)
Y_random_mean = np.sign(X2 - a_random_mean)
Y_svm_mean = np.sign(X2 - a_svm_mean)
#计算平均值
bias_random = np.mean(Y != Y_random_mean)
bias_svm = np.mean(Y != Y_svm_mean)
print("bias_random = {}".format(bias_random))
print("bias_svm = {}".format(bias_svm))

#计算var_random_mean,var_svm
var_random = np.array([])
var_svm = np.array([])
for i in range(1000):
    #生成数据
    X1p, X2p, X1n, X2n = generate()
    #计算随机选择以及svm算法对应的系数
    a_random = np.random.uniform(-1, 1)
    a_svm = mysvm(X2p, X2n)
    #生成用于模拟的数据
    X2 = np.random.uniform(-1, 1, 1000)
    #计算标签
    Y_random = np.sign(X2 - a_random)
    Y_svm = np.sign(X2 - a_svm)
    #计算平均值
    Y_random_mean = np.sign(X2 - a_random_mean)
    Y_svm_mean = np.sign(X2 - a_svm_mean)
    #计算样本方差
    var_random = np.append(var_random, np.mean(Y_random_mean != Y_random))
    var_svm = np.append(var_svm, np.mean(Y_svm_mean != Y_svm))
    
print("var_svm = {}".format(np.mean(var_svm)))
print("var_random = {}".format(np.mean(var_random)))