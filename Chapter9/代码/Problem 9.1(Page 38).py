# -*- coding: utf-8 -*-
"""
Created on Thu May  2 09:25:37 2019

@author: qinzhen
"""

import numpy as np
import helper as hlp

X = np.array([[0, 0], [0, 1], [5, 5]]).astype("float")
y = np.array([1, 1, -1])
#(a)
knn = hlp.KNeighborsClassifier_(1)
knn.fit(X, y)
hlp.draw(X, y, knn)

#(b)
scaler = hlp.StandardScaler_()
X1 = scaler.fit_transform(X)
knn = hlp.KNeighborsClassifier_(1)
knn.fit(X1, y)

hlp.draw(X, y, knn, flag=2, preprocess=scaler)

#(c)
pca = hlp.PCA_(1)
pca.fit(X)
X2 = pca.transform(X)
knn = hlp.KNeighborsClassifier_(1)
knn.fit(X2, y)

hlp.draw(X, y, knn, flag=3, preprocess=pca)