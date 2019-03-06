# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 09:50:23 2019

@author: qinzhen
"""

#Step1 构造损失函数
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

def Class(s,y):
    if s * y > 0:
        return 0
    else:
        return 1
    
def Sq(s, y):
    return (s - y) ** 2

def Log(s, y):
    return np.log(1 + np.exp(- y * s))

#Step2 构造点集并作图
#构造点
x = np.arange(-2,2,0.01)

#y=1
eclass1 = [Class(i, 1) for i in x]
esq1 = [Sq(i, 1) for i in x]
elog1 = [Log(i, 1) / np.log(2) for i in x]

#y=-1
eclass2 = [Class(i, -1) for i in x]
esq2 = [Sq(i, -1) for i in x]
elog2 = [Log(i, -1) / np.log(2) for i in x]

plt.plot(x, eclass1, label='eclass')
plt.plot(x, esq1, label='esq')
plt.plot(x, elog1, label='elog')
plt.title('y=1')
plt.legend()
plt.show()

plt.plot(x, eclass2, label='eclass')
plt.plot(x, esq2, label='esq')
plt.plot(x, elog2, label='elog')
plt.title('y=-1')
plt.legend()
plt.show()