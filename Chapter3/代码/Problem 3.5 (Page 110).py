# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 12:02:33 2019

@author: qinzhen
"""

import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

def f1(s):
    a = max(0, 1 - s)
    return a

def f2(s):
    if s > 0:
        return 0
    else:
        return 1
    
x = np.linspace(-2, 2, 500)
y1 = [f1(i) for i in x]
y2 = [f2(i) for i in x]

plt.plot(x ,y1, label="(max(0,1-s))**2")
plt.plot(x, y2, label="[sign(s)!=1]")
plt.legend()
plt.title('Ein的比较')
plt.show()