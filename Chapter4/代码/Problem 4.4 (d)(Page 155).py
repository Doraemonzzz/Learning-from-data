# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 17:53:57 2019

@author: qinzhen
"""

import numpy as np
import pickle
from numpy.linalg import inv
from scipy.integrate import quad
from sklearn.preprocessing import PolynomialFeatures
from helper import process
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号


#############################
##stochastic noise

E = {}
t = 100

Qf = [20]
N = np.arange(5, 122, 5)
sigma2 = np.arange(0, 2.01, 0.05)
q = 20
'''
#进行实验，因为时间太长，这里注释掉
for n in N:
    for s in sigma2:
        e = np.array([])
        for i in range(t):
            error = process(q, n, s)
            e = np.append(e, error)
        E[(q, n, s)] = np.mean(e)

#保存数据
#with open('stochastic noise.pickle', 'wb') as f:
#    pickle.dump(E, f)
'''

#如果不训练读取数据即可
#读取数据
pickle_in = open('stochastic noise.pickle', 'rb')
E = pickle.load(pickle_in)


#读取后横坐标N以及纵坐标sigma2以及对应的差值c      
c = []
n1 = []
s1 = []
for i in E:
    n1.append(i[1])
    s1.append(i[2])
    c.append(E[i])

c = np.array(c)
n1 = np.array(n1)
s1 = np.array(s1)

n1, s1 = np.meshgrid(n1, s1)
result = []
for i in range(len(n1)):
    n = n1[i]
    s = s1[i]
    temp = []
    for j in range(len(n)):
        temp.append(E[(q, n[j], s[j])])
    result.append(temp)

cm = plt.cm.get_cmap('rainbow')
plt.pcolormesh(n1, s1, result, cmap=cm, vmin=-0.2, vmax=0.2, shading='gouraud', edgecolors='face')
plt.xlabel("Number of Data Points, N")
plt.ylabel("Noise Level,sigma^2")
plt.title("stochastic noise")
plt.colorbar()
plt.show()

#############################
##deterministic noise

Qf = np.arange(1, 31)
N = np.arange(5, 121, 5)
sigma2 = 0.1
s = sigma2

E = {}
t = 150
'''
#进行实验，因为时间太长，这里注释掉
for q in Qf:
    for n in N:
        e = np.array([])
        for i in range(t):
            error = process(q, n, s)
            e = np.append(e, error)
        E[(q,n,s)] = np.mean(e)

#保存数据
#with open('deterministic noise.pickle', 'wb') as f:
#    pickle.dump(E, f)
'''

#如果不训练读取数据即可
#读取数据
pickle_in = open('deterministic noise.pickle', 'rb')
E = pickle.load(pickle_in)

c = []
q1 = []
n1 = []
for i in E:
    q1.append(i[0])
    n1.append(i[1])
    c.append(E[i])

c = np.array(c)
q1 = np.array(q1)
n1 = np.array(n1)

n1, q1 = np.meshgrid(n1, q1)
result = []

for i in range(len(q1)):
    q = q1[i]
    n = n1[i]
    temp = []
    for j in range(len(q)):
        temp.append(E[(q[j], n[j], sigma2)])
    result.append(temp)
   
cm = plt.cm.get_cmap('rainbow')
plt.pcolormesh(n1, q1, result, cmap=cm, vmin=-0.2, vmax=0.2, shading='gouraud', edgecolors='face')
plt.colorbar()
plt.xlabel("Number of Data Points, N")
plt.ylabel("Target Complexity, Qf")
plt.title("deterministic noise")
plt.show()
