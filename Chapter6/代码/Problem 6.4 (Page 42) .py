# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 14:16:19 2019

@author: qinzhen
"""

import helper as hlp

#Step1 产生数据
#参数
rad = 10
thk = 5
sep = 5

#产生数据
X, y = hlp.generatedata(rad, thk, sep, 100)

knn = hlp.KNeighborsClassifier_(1)
knn.fit(X, y)
hlp.draw(X, y, knn)

knn = hlp.KNeighborsClassifier_(3)
knn.fit(X, y)
hlp.draw(X, y, knn)