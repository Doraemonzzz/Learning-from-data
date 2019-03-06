# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 10:44:29 2019

@author: qinzhen
"""

import numpy as np
from numpy.linalg import inv
from helper import generatedata
from helper import Pocket_PLA
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

#Step1 产生数据
#参数
rad = 10
thk = 5
sep = -5
N = 2000

#产生数据
X, y = generatedata(rad, thk, sep, N)

#作图
plt.scatter(X[y>0][:, 0], X[y>0][:, 1], s=1)
plt.scatter(X[y<0][:, 0], X[y<0][:, 1], s=1)
plt.show()

#Step2 训练数据
#Pocket_PLA
#对数据预处理，加上偏置项项1
X_treat = np.c_[np.ones(N), X]

#迭代次数
max_step = 10000

#产生结果
W, W_hat, w, error = Pocket_PLA(X_treat, y, max_step=max_step)
ein = np.mean(np.sign(W_hat.dot(X_treat.T)) != y, axis=1)

#(b)
t = np.arange(max_step)
plt.plot(t, ein)
plt.title('Ein VS t')
plt.show()

#(c)
r = 2 * (rad + thk)
a1 = np.array([-r,r])
b1 = - (w[0] + w[1] * a1) / w[2]

plt.scatter(X[y>0][:, 0], X[y>0][:, 1], s=1)
plt.scatter(X[y<0][:, 0], X[y<0][:, 1], s=1)
plt.plot(a1, b1)
plt.title('Pocket PLA')
plt.show()
print('Pocket PLA的错误率为' + str(error / N))

#(d)
#Linear regression
w_lr = inv(X_treat.T.dot(X_treat)).dot(X_treat.T).dot(y)
 
#作图
a2 = np.array([-r,r])
b2 = - (w_lr[0] + w_lr[1] * a1) / w_lr[2]
 
plt.scatter(X[y>0][:, 0], X[y>0][:, 1], s=1)
plt.scatter(X[y<0][:, 0], X[y<0][:, 1], s=1)
plt.plot(a2, b2)
plt.title('linear regression')
plt.show()
error = np.mean(np.sign(X_treat.dot(w_lr)) != y)
print('linear regression的错误率为' + str(error))

#(e)
#特征转换
poly = PolynomialFeatures(3)
X_poly = poly.fit_transform(X)

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

#数据数目
n = 2000
#定义a, b
a = np.linspace(-r, r, n)
b = np.linspace(-r, r, n)

#生成网格数据
A, B = np.meshgrid(a, b)

#迭代次数
max_step = 10000

#Pocket_PLA
W_poly, W_poly_hat, w_poly, error_poly = Pocket_PLA(X_poly, y, max_step=max_step)
ein_poly = np.mean(np.sign(W_poly_hat.dot(X_poly.T)) != y, axis=1)

#(b)
plt.plot(t, ein_poly)
plt.title('Ein VS t')
plt.show()

#(c)
plt.contour(A, B, f(A, B, w_poly), 1, colors = 'red')
plt.scatter(X[y>0][:, 0], X[y>0][:, 1], s=1)
plt.scatter(X[y<0][:, 0], X[y<0][:, 1], s=1)
plt.title('Pocket PLA')
plt.show()
print('特征转换后的Pocket PLA的错误率为' + str(error_poly / N))

#(d)Linear regression
w_poly_lr = inv(X_poly.T.dot(X_poly)).dot(X_poly.T).dot(y)
plt.contour(A, B, f(A, B, w_poly_lr), 1, colors = 'red')
plt.scatter(X[y>0][:, 0], X[y>0][:, 1], s=1)
plt.scatter(X[y<0][:, 0], X[y<0][:, 1], s=1)
plt.title('Pocket PLA')
plt.show()
error = np.mean(np.sign(X_poly.dot(w_poly_lr)) != y)
print('特征转换后的linear regression的错误率为' + str(error))
