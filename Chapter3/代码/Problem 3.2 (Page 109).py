# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 10:38:42 2019

@author: qinzhen
"""

import numpy as np
from helper import generatedata
from helper import PLA
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号


#参数
rad = 10
thk = 5
Sep = np.arange(0.2,5.2,0.2)
N = 2000
#实验次数
n = 30


#记录迭代次数
T = np.array([])

for sep in Sep:
    t1 = 0
    for i in range(n):
        X, y = generatedata(rad, thk, sep, N)
        X_treat = np.c_[np.ones(N), X]
        
        t, last, w = PLA(X_treat, y)
        t1 += t
    
    T = np.append(T, t1 / n)
    
plt.plot(Sep, T)
plt.title('sep和迭代次数的关系')
plt.show()