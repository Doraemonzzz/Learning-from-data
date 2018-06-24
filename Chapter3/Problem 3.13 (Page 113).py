# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 09:07:51 2018

@author: Administrator
"""

import numpy as np
import matplotlib.pylab as plt
from numpy.linalg import inv

#(c)
def generate(n,delta):
    X=[]
    Y=[]
    data=[]
    for i in range(n):
        x=np.random.random()
        t=np.random.normal(0,1)
        y=x*x+delta*t
        data.append([x,y])
        X.append(x)
        Y.append(y)
    return X,Y,data

n=50
delta=0.1

X,Y,data=generate(n,delta)

a=np.array([0,0.1])
D1=np.array(data)+a
D2=np.array(data)-a

X1=D1[:,0]
Y1=D1[:,1]
X2=D2[:,0]
Y2=D2[:,1]

plt.scatter(X1,Y1,label='+a')
plt.scatter(X2,Y2,label='-a')
plt.legend()
plt.show()


#(d)
#线性回归
X1=np.array([[1,i] for i in X])
Y=np.array(Y)
w1=inv(X1.T.dot(X1)).dot(X1.T).dot(Y)

x=[0,1]
y=[w1[0]+w1[1]*i for i in x]

plt.scatter(X,Y)
plt.plot(x,y,'r')
plt.show()

#此题介绍的分类方法
#构造数据
a=np.array([0,0.1])
data=np.array(data)
D=[]
for i in data:
    t1=i+a
    t1=np.array([1]+list(t1)+[1])
    t2=i-a
    t2=np.array([1]+list(t2)+[-1])
    D.append(t1)
    D.append(t2)
    
from cvxopt import matrix, solvers

finaldata=[]
for i in D:
    temp=list(i[:-1]*i[-1])
    finaldata.append(temp)

A=np.array(finaldata)*(-1)

n=len(finaldata)
m=len(finaldata[0])
    
#根据之前所述构造矩阵
a1=np.concatenate((A,(-1)*np.eye(n)),axis=1)
a2=np.concatenate((np.zeros([n,m]),(-1)*np.eye(n)),axis=1)

A1=np.concatenate((a1,a2))
c1=np.array([0.0]*m+[1.0]*n)
b1=np.array([-1.0]*n+[0.0]*n)

#带入算法求解
c1=matrix(c1)
A1=matrix(A1)
b1=matrix(b1)

sol1 = solvers.lp(c1,A1,b1)

w2=list(sol1['x'])

x1=[0,1]
y1=[-(w2[0]+w2[1]*i)/w2[2] for i in x1]

plt.scatter(X,Y)
plt.plot(x,y,'r')
plt.show()
