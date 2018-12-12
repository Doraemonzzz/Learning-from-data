# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 11:46:58 2018

@author: Administrator
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import decomposition

data = np.genfromtxt("zip.train")
X = data[:, 1:]
# center
mean = np.mean(X, axis=0)
X = X - mean
Y = data[:, 0]
l = 1
X1 = X[Y == l]
X2 = X[Y != l]

pca = decomposition.PCA(n_components=1)
pca.fit(X1)
z1 = pca.transform(X)
pca.fit(X2)
z2 = pca.transform(X)

plt.scatter(z1[Y == l], z2[Y == l], s=5)
plt.scatter(z1[Y != l], z2[Y != l], s=5)
plt.title("+1 digits VS -1 digits with 2 components PCA")
plt.show()