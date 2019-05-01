# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 12:48:23 2019

@author: qinzhen
"""

#(a)
import numpy as np
import helper as hlp

X = np.array([[1, 0], [0, 1], [0, -1], [-1, 0], [0, 2], [0, -2], [-2, 0]])
y = np.array([-1, -1, -1, -1, 1, 1, 1])

knn = hlp.KNeighborsClassifier_(1)
knn.fit(X, y)
hlp.draw(X, y, knn)

knn = hlp.KNeighborsClassifier_(3)
knn.fit(X, y)
hlp.draw(X, y, knn)

#(b)
#特征转换
Z = np.c_[np.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2), np.arctan(X[:, 1] / (X[:, 0]  + 10**(-8)))]
knn = hlp.KNeighborsClassifier_(1)
knn.fit(Z, y)
hlp.draw(X, y, knn, flag=0)

knn = hlp.KNeighborsClassifier_(3)
knn.fit(Z, y)
hlp.draw(X, y, knn, flag=0)