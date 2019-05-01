# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 14:00:50 2019

@author: qinzhen
"""

import numpy as np
import helper as hlp
    
X = np.array([[1, 0], [0, 1], [0, -1], [-1, 0], [0, 2], [0, -2], [-2, 0]])
y = np.array([-1, -1, -1, -1, 1, 1, 1])

#(a)
X1 = np.array([[np.mean(X[y>0][:, 0]), np.mean(X[y>0][:, 1])], [np.mean(X[y<0][:, 0]), np.mean(X[y<0][:, 1])]])
y1 = np.array([1, -1])

knn = hlp.KNeighborsClassifier_(1)
knn.fit(X1, y1)
hlp.draw(X, y, knn)

#(b)
def f(X):
    while(len(X) >1):
        #记录当前距离
        d = float('inf')
        #元素数量
        n = len(X)
        #记录最优元素的下标
        k = 0
        l = 0
        for i in range(n):
            for j in range(i+1, n):
                d1 = np.sum((X[i] - X[j])**2)
                if(d > d1):
                    d = d1
                    k = i
                    l = j
        #生成新的元素
        data = (X[k] + X[l]) / 2
        #删除元素
        X = np.delete(X, l, axis=0)
        X = np.delete(X, k, axis=0)
        #增加新元素
        X = np.append(X, data.reshape(-1, 2), axis=0)
    return X[0]

#划分数据
X_pos = X[y>0]
X_neg = X[y<0]

#新数据
X2 = np.array([f(X_pos), f(X_neg)])
y2 = np.array([1, -1])

knn = hlp.KNeighborsClassifier_(1)
knn.fit(X2, y2)
hlp.draw(X, y, knn)

