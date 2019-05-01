# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 16:15:56 2019

@author: qinzhen
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import integrate
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

def f(x):
    return math.sin(math.pi * x)


#### Part 1: y=kx+b
#已知(x1,y1),(x2,y2)过直线y=kx+b，此函数返回k,b
def treat(x1,x2):
    y1 = f(x1)
    y2 = f(x2)
    k = (y2 - y1) / (x2 - x1)
    b = (y1 * x2 - y2 * x1) / (x2 - x1)
    return k, b

#以0.05为间隔产生-1到1的点集，对每个点，求每个点与其后面一个点连线的直线
x = np.arange(-1, 1, 0.05)

#求出对应的k,b
u = []
for i in range(x.size-1):
    u.append(treat(x[i], x[i+1]))

#求出直线
X1 = np.array([-1.0, 1.0])
Y1 = []
for i in u:
    temp = X1 * i[0] + i[1]
    Y1.append(temp)

#y=sin(pix)
X2 = np.arange(-1,1,0.01)
Y2 = np.sin(np.pi * X2)

#作图
for i in Y1:
    plt.plot(X1, i)
plt.scatter(X2, Y2, c='BLACK')
plt.show()

#定义误差函数
def s1(x, x1, x2):
    #此处用numpy计算会产生错误，暂时没找到原因
    y1 = f(x1)
    y2 = f(x2)
    #为了防止分母为0
    try:
        k = (y2 - y1) / (x2 - x1)
        b = (y1 * x2 - y2 * x1) / (x2 - x1)
    except:
        k = (y2 - y1) / (x2 - x1 + 10 ** (-10))
        b = (y1 * x2 - y2 * x1) / (x2 - x1 + 10 ** (-10))
    y = f(x)
    y0 = k * x + b
    return 1 / 8 * (y - y0) ** 2

print(integrate.tplquad(s1, -1, 1, lambda x: -1, lambda x: 1,lambda x, y: -1, lambda x, y: 1))

#计算bias,var
#计算bias
#模拟求出系数
def simulation(n):
    k = 0
    b = 0
    #模拟n次
    for i in range(n):
        #产生-1,1之间均匀分布随机数
        x1 = np.random.uniform(-1, 1)
        x2 = np.random.uniform(-1, 1)
        k1, b1 = treat(x1, x2)
        k += k1
        b += b1
    return k / n, b / n

#比较下10000次以上的结果
n = range(10 ** 4, 10 ** 5+1, 10 ** 4)
result =[]
for i in n:
    temp = simulation(i)
    print(temp)
    result.append(temp)

#取第一组计算bias
a1, b1 = result[0]

def bias(x):
    y1 = a1 * x + b1
    y2 = f(x)
    return (y1 - y2) ** 2 / 2

print(integrate.quad(bias,-1,1))

#计算var
def var(x, x1, x2):
    y1 = f(x1)
    y2 = f(x2)
    try:
        k = (y2 - y1) / (x2 - x1)
        b = (y1 * x2 - y2 * x1) / (x2 - x1)
    except:
        k = (y2 - y1) / (x2 - x1 + 10 ** (-10))
        b = (y1 * x2 - y2 * x1) / (x2 - x1 + 10 ** (-10))
    #之前计算的平均系数
    yavg= a1 * x + b1
    #真实值
    yrea = k * x + b
    return 1 / 8 * (yavg - yrea) ** 2

print(integrate.tplquad(var, -1, 1, lambda x: -1, lambda x: 1,lambda x, y: -1, lambda x, y: 1))


#### Part 2: y=kx
#定义误差函数
def s1_b(x, x1, x2):
    y1 = f(x1)
    y2 = f(x2)
    #为了防止分母为0
    try:
        a = (x1 * y1 + x2 * y2) / (x1 ** 2 + x2 ** 2)
    except:
        a = (x1 * y1 + x2 * y2) / (x1 ** 2 + x2 ** 2 + 10 ** (-6))
    y = f(x)
    y0 = a*x
    return 1 / 8 * (y - y0) ** 2

print(integrate.tplquad(s1_b, -1, 1, lambda x: -1, lambda x: 1,lambda x, y: -1, lambda x, y: 1))

#bias
#模拟求出系数
def simulation_b(n):
    a = 0
    #模拟n次
    for i in range(n):
        #产生-1,1之间均匀分布随机数
        x1 = np.random.uniform(-1, 1)
        x2 = np.random.uniform(-1, 1)
        y1 = f(x1)
        y2 = f(x2)
        a1 = (x1 * y1 + x2 * y2) / (x1 ** 2 + x2 ** 2)
        a += a1
    return a / n

#比较下10000次以上的结果
n = range(10 ** 4, 10 ** 5+1, 10 ** 4)
result =[]
for i in n:
    temp = simulation_b(i)
    print(temp)
    result.append(temp)

#取第一组计算bias
a1 = result[0]
#bias
def bias_b(x):
    y1 = a1 * x
    y2 = f(x)
    return (y1 - y2) ** 2 / 2

print(integrate.quad(bias_b, -1, 1))

#var
def var_b(x, x1, x2):
    y1 = f(x1)
    y2 = f(x2)
    try:
        a = (x1 * y1 + x2 * y2) / (x1 ** 2 + x2 ** 2)
    except:
        a = (x1 * y1 + x2 * y2) / (x1 ** 2 + x2 ** 2 + 10 ** (-6))
    yavg = a1 * x
    yrea = a * x
    return 1 / 8 * (yavg - yrea) ** 2

print(integrate.tplquad(var_b, -1, 1, lambda x: -1, lambda x: 1,lambda x, y: -1, lambda x, y: 1))


#### Part 3: y=b
#定义误差函数
def s1_c(x, x1, x2):
    y1 = f(x1)
    y2 = f(x2)
    b = (y1 + y2) / 2
    y = f(x)
    y1 = b
    return 1 / 8 * (y - y1) ** 2
print(integrate.tplquad(s1_c, -1, 1, lambda x: -1, lambda x: 1,lambda x, y: -1, lambda x, y: 1))

#bias
#模拟求出系数
def simulation_c(n):
    b = 0
    #模拟n次
    for i in range(n):
        #产生-1,1之间均匀分布随机数
        x1 = np.random.uniform(-1, 1)
        x2 = np.random.uniform(-1, 1)
        y1 = f(x1)
        y2 = f(x2)
        b1 = (y1 + y2) / 2
        b += b1
    return(b / n)

#比较下10000次以上的结果
n = range(10 ** 4, 10 ** 5+1, 10 ** 4)
result =[]
for i in n:
    temp = simulation_c(i)
    print(temp)
    result.append(temp)
    
#取第一组计算bias
b1 = result[0]
def bias_c(x):
    y1 = b1
    y2 = f(x)
    return (y1 - y2) ** 2 / 2

print(integrate.quad(bias_c, -1, 1))

#var
def var_c(x, x1, x2):
    y1 = f(x1)
    y2 = f(x2)
    b = (y1 + y2)/2
    yavg = b1
    yrea = b 
    return 1 / 8 * (yavg - yrea) ** 2

print(integrate.tplquad(var_c, -1, 1, lambda x: -1, lambda x: 1,lambda x, y: -1, lambda x, y: 1))