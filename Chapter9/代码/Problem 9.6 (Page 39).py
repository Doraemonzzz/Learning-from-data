# -*- coding: utf-8 -*-
"""
Created on Thu May  2 11:08:48 2019

@author: qinzhen
"""

import numpy as np
import matplotlib.pyplot as plt
import helper as hlp

data = np.genfromtxt("zip.train")
X = data[:, 1:]
y = data[:, 0]
#PCA分解with centering
pca = hlp.PCA_(n_components=2)
X_pca = pca.fit_transform(X)

plt.scatter(X_pca[:, 0], X_pca[:, 1], s=5, c=y, cmap='gist_ncar')
plt.colorbar()
plt.title("digits data with 2 components PCA with centering")
plt.show()

#PCA分解without centering
pca = hlp.PCA_(n_components=2)
X_pca = pca.fit_transform(X, flag=2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], s=5, c=y, cmap='gist_ncar')
plt.colorbar()
plt.title("digits data with 2 components PCA without centering")
plt.show()