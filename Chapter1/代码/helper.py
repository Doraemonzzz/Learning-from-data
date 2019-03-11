# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 12:01:52 2019

@author: qinzhen
"""

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

def Judge(X, y, w):
    """
    判别函数，判断所有数据是否分类完成
    """
    n = X.shape[0]
    #判断是否同号
    num = np.sum(X.dot(w) * y > 0)
    return num == n

def data(N, d, rnd, t=0.1):
    """
    生成N个d维点（不包括偏置项1），x1+...+xd>=t的点标记为+1，x1+...+xd<=-t的点标记为-1，
	rnd为随机数生成器，形式为rnd = np.random.RandomState(seed)，seed为随机种子
    """
    X = []
    w = np.ones(d)
    while (len(X) < N):
        x = rnd.uniform(-1, 1, size=(d))
        if np.abs(x.dot(w)) >= t:
            X.append(x)
    
    X = np.array(X)
    y = 2 * (X.dot(w) > 0) - 1
    #添加第一个分量为1
    X = np.c_[np.ones((N, 1)), X]
    
    return X, y

def f(N, d, rnd, t=0.1, r=1):
    """
    生成N个d维点（不包括偏置项1），x1+...+xd>=t的点标记为+1，x1+...+xd<=-t的点标记为-1，
    rnd为随机数生成器，形式为rnd = np.random.RandomState(seed)，seed为随机种子
	利用PLA更新，如果r=1，那么按照顺序取点，否则随机取点
    """
    X, y = data(N, d, rnd, t=t)
    
    #记录次数
    s = 0
    #初始化w=[0, 0, 0]
    w = np.zeros(d + 1)
    #数据数量
    n = X.shape[0]
    if r == 1:
        while (Judge(X, y ,w) == 0):
            for i in range(n):
                if X[i, :].dot(w) * y[i] <= 0:
                    w += y[i] * X[i, :]
                    s += 1
    else:
        while (Judge(X, y ,w) == 0):
            i = np.random.randint(0, N)
            if X[i, :].dot(w) * y[i] <= 0:
                w += y[i] * X[i, :]
                s += 1
                
    #直线方程为w0+w1*x1+w2*x2=0,根据此生成点
    a = np.arange(-1, 1, 0.1)
    b = (a * w[1] + w[0]) / (- w[2])
    
    #原直线方程为x1+x2 = 0
    c = - a
    
    #返回数据
    return a, b, c, X, y, s, w

def plot_helper(a, b, c, X, y, s, w, t=0):
    """
    作图函数
    """
    #画出图像
    plt.scatter(X[y == 1][:, 1], X[y == 1][:, 2], c='r', s=1)
    plt.scatter(X[y == -1][:, 1], X[y == -1][:, 2], c='b', s=1)
    plt.plot(a, b, label="("+str(w[0])+")+("+str(w[1])+")x1+("+str(w[2])+")x2=0")
    plt.plot(a, c, label="x1+x2="+str(t))
    plt.title(u"经过"+str(s)+u"次迭代收敛")
    plt.legend()
    plt.show()