# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 10:59:02 2019

@author: qinzhen
"""

import numpy as np
import matplotlib.pylab as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
from helper import Pocket_PLA
from helper import preprocess

#Step1产生数据
def generatedata(n, flag=True):
    """
    产生n组数据，10%为噪声，这里直线选择为y=x，y>x则标签为1，否则为-1
    x, y的范围都介于(-2, 2)，flag用于判断是否产生噪声
    """
    #产生X
    X = np.random.uniform(-2, 2, (n, 2))
    #计算标签
    y = np.ones(n)
    y[X[:, 0] <= X[:, 1]] = -1
    if flag:
        #让前10%数据误分
        y[: n//10] *= -1
    #合并数据
    Data = np.c_[X, y]
    #打乱数据
    np.random.shuffle(Data)
    
    return Data

#Step2展示数据
#生成训练数据
D_train = generatedata(100)
X_train = D_train[:, :-1]
y_train = D_train[:, -1]
#后面两步是为了作图
#标签为+1的点
train_px = X_train[y_train > 0][:, 0]
train_py = X_train[y_train > 0][:, 1]
#标签为-1的点
train_nx = X_train[y_train < 0][:, 0]
train_ny = X_train[y_train < 0][:, 1]


#生成测试数据
D_test = generatedata(1000)
X_test = D_test[:, :-1]
y_test = D_test[:, -1]
#标签为+1的点
test_px = X_test[y_test > 0][:, 0]
test_py = X_test[y_test > 0][:, 1]
#标签为-1的点
test_nx = X_test[y_test < 0][:, 0]
test_ny = X_test[y_test < 0][:, 1]

x = [-2, 2]
y = [-2, 2]

#训练数据
plt.scatter(train_px, train_py, s=1)
plt.scatter(train_nx, train_ny, s=1)
plt.title('训练数据')
plt.plot(x, y, label='y=x')
plt.legend()
plt.show()

#测试数据
plt.scatter(test_px, test_py, s=1)
plt.scatter(test_nx, test_ny, s=1)
plt.title('测试数据')
plt.plot(x, y, label='y=x')
plt.legend()
plt.show()


#Step3训练数据
#n为迭代次数，k为步长，N为实验次数
n = 1000
k = 1
N = 20

def experiment(n, k):
    """
    模拟一次实验，n为迭代次数，k为步长
    """
    #训练数据
    D_train = generatedata(100)
    X_train, y_train = preprocess(D_train)
    
    #测试数据
    D_test = generatedata(1000)
    X_test, y_test = preprocess(D_test)
    
    #训练模型
    W, W_hat, w_hat, error = Pocket_PLA(X_train, y_train, k, n)
    
    #计算错误率
    ein = np.mean(np.sign(W.dot(X_train.T)) != y_train, axis=1)
    ein_hat = np.mean(np.sign(W_hat.dot(X_train.T)) != y_train, axis=1)
    eout = np.mean(np.sign(W.dot(X_test.T)) != y_test, axis=1)
    eout_hat = np.mean(np.sign(W_hat.dot(X_test.T)) != y_test, axis=1)
    return ein, ein_hat, eout, eout_hat

#存储结果
Ein = np.zeros(n)
Ein_hat = np.zeros(n)
Eout = np.zeros(n)
Eout_hat = np.zeros(n)

for i in range(N):
    ein, ein_hat, eout, eout_hat = experiment(n, k)
    Ein += ein
    Ein_hat += ein_hat
    Eout += eout
    Eout_hat += eout_hat
    
#计算均值
Ein /= N
Ein_hat /= N
Eout /= N
Eout_hat /= N

t = np.arange(1, n+1)

#Ein作图
plt.plot(t, Ein, label='Ein')
plt.plot(t, Ein_hat, label='Ein_hat')
plt.title('Ein')
plt.xlabel('训练次数')
plt.ylabel('错误率')
plt.legend()
plt.show()

#Eout作图
plt.plot(t, Eout, label='Eout')
plt.plot(t, Eout_hat, label='Eout_hat')
plt.title('Eout')
plt.xlabel('训练次数')
plt.ylabel('错误率')
plt.legend()
plt.show()