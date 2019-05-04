# -*- coding: utf-8 -*-
"""
Created on Thu May  2 14:50:33 2019

@author: qinzhen
"""

import numpy as np
import matplotlib.pyplot as plt
import helper as hlp

data = np.genfromtxt("zip.train")
X = data[:, 1:]
#中心化
X -= np.mean(X, axis=0)
y = data[:, 0]
#选择的标签
l = 1
#划分数据
Xpos = X[y == l]
Xneg = X[y != l]
#对每一类分别使用PCA，flag=2表示pca中不使用中心化
pca = hlp.PCA_(n_components=1)
pca.fit(Xpos, flag=2)
z1 = pca.transform(X)

pca = hlp.PCA_(n_components=1)
pca.fit(Xneg, flag=2)
z2 = pca.transform(X)

#作图
plt.scatter(z1[y == l], z2[y == l], s=5)
plt.scatter(z1[y != l], z2[y != l], s=5)
plt.title("+1 digits VS -1 digits with 2 components PCA")
plt.show()