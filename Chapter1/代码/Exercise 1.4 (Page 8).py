# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 21:32:17 2019

@author: qinzhen
"""

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

#设置随机种子，保证每次结果一致
seed = 42
rnd = np.random.RandomState(42)

#首先生成20个可分的数据，为方便起见，分别生成10个第一象限的点和10个第三象限的点
#第一象限10个点
X1 = rnd.uniform(0, 1, size=(10, 2))
y1 = np.ones(10)
#第三象限10个点
X2 = rnd.uniform(-1, 0, size=(10, 2))
y2 = -1 * np.ones(10)

#按行拼接
X = np.r_[X1, X2]
y = np.r_[y1, y2]
#添加第一个分量为1
X = np.c_[np.ones((20, 1)), X]

#定义判别函数，判断所有数据是否分类完成
def Judge(X, y, w):
    flag = 1
    n = X.shape[0]
    for i in range(n):
        if X[i, :].dot(w) * y[i] <= 0:
            flag = 0
            break
    return flag

#记录次数
s = 0
#初始化w=[0, 0, 0]
w = np.array([0,0,0], dtype=float)
#数据数量
n = X.shape[0]
while (Judge(X, y ,w) == 0):
    for i in range(n):
        if X[i, :].dot(w) * y[i] <= 0:
            w += y[i] * X[i, :]
            s += 1
            
#直线方程为w0+w1*a+w2*b=0,根据此生成点
a = np.arange(-1, 1, 0.1)
b = np.array([(i * w[1] + w[0]) / (-w[2]) for i in a])

#画出图片
plt.scatter(X1[:, 0], X1[:, 1], c='r')
plt.scatter(X2[:, 0], X2[:, 1], c='b')
plt.plot(a, b)
plt.title(u"经过"+str(s)+u"次迭代收敛")
plt.show()