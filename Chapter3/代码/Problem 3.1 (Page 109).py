# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 10:18:23 2019

@author: qinzhen
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from helper import generatedata
from helper import PLA

#Step1 产生数据
#参数
rad = 10
thk = 5
sep = 5
N = 2000

#产生数据
X, y = generatedata(rad, thk, sep, N)

#作图
plt.scatter(X[y>0][:, 0], X[y>0][:, 1], s=1)
plt.scatter(X[y<0][:, 0], X[y<0][:, 1], s=1)
plt.show()


#Step2 训练数据
#(a)PLA
#对数据预处理，加上偏置项项1
X_treat = np.c_[np.ones(N), X]

#PLA
t, last, w = PLA(X_treat, y)

#作出直线
r = 2 * (rad + thk)
a1 = np.array([-r,r])
b1 = - (w[0] + w[1] * a1) / w[2]

plt.scatter(X[y>0][:, 0], X[y>0][:, 1], s=1)
plt.scatter(X[y<0][:, 0], X[y<0][:, 1], s=1)
plt.plot(a1, b1, c="red")
plt.title('PLA')
plt.show()

#(b)linear regression
w1 = inv(X_treat.T.dot(X_treat)).dot(X_treat.T).dot(y)

#作图
a2 = np.array([-r,r])
b2 = - (w1[0] + w1[1] * a1) / w1[2]

plt.scatter(X[y>0][:, 0], X[y>0][:, 1], s=1)
plt.scatter(X[y<0][:, 0], X[y<0][:, 1], s=1)
plt.plot(a2, b2, c="red")
plt.title('linear regression')
plt.show()