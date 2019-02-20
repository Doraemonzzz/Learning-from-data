# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 17:23:17 2019

@author: qinzhen
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

def f(x):
    return x ** 2

#模拟求出系数
def simulation(n):
    a = 0
    b = 0
    #模拟n次
    for i in range(n):
        #产生-1,1之间均匀分布随机数
        x1 = np.random.uniform(-1, 1)
        x2 = np.random.uniform(-1, 1)
        a1 = (x1 + x2)
        b1 = - x1 * x2
        a += a1
        b += b1
    return a / n, b / n

#比较下10000次以上的结果
n = range(10 ** 4, 10 ** 5+1, 10 ** 4)
result =[]
for i in n:
    temp = simulation(i)
    print(temp)
    result.append(temp)

#选择第一组数据并作图
a, b = result[0]
x1 = np.arange(-1, 1.1, 0.1)
y1 = [f(i) for i in x1]
x2 = np.array([-1, 1])
y2 = a * x2 + b

plt.plot(x1,y1)
plt.plot(x2,y2)
plt.show()

#计算积分
#定义误差函数
def s1(x, x1, x2):
    a = (x1 + x2)
    b = - x1 * x2
    y = f(x)
    y1 = a * x + b
    return 1 / 8 * (y - y1) ** 2

print(integrate.tplquad(s1, -1, 1, lambda x: -1, lambda x: 1,lambda x, y: -1, lambda x, y: 1))

#bias
def bias(x):
    y1 = a * x + b
    y2 = f(x)
    return (y1 - y2) ** 2 / 2

print(integrate.quad(bias, -1, 1))

#var
def var(x, x1, x2):
    yavg = a * x + b
    a1 = (x1 + x2)
    b1 = -x1 * x2
    yrea = a1 * x + b1
    return 1 / 8 * (yavg - yrea) ** 2

print(integrate.tplquad(var, -1, 1, lambda x: -1, lambda x: 1,lambda x, y: -1, lambda x, y: 1))
