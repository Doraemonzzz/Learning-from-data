# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 11:12:22 2018

@author: Administrator
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import decomposition

data = np.genfromtxt("zip.train")
X = data[:, 1:]
Y = data[:, 0]
pca = decomposition.PCA(n_components=2)
pca.fit(X)
X_pca = pca.transform(X)

plt.scatter(X_pca[:, 0], X_pca[:, 1], s=5, c = Y, cmap='gist_ncar')
plt.colorbar()
plt.title("digits data with 2 components PCA")
plt.show()