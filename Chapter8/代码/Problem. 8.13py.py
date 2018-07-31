# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 11:41:16 2018

@author: Administrator
"""

import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
from numpy.linalg import inv

####a

X = np.array([(-0.494, 0.363),(-0.311, -0.101),(-0.0064, 0.374),(-0.0089, -0.173),
              (0.0014, 0.138),(-0.189, 0.718),(0.085, 0.32208),(0.171, -0.302),(0.142, 0.568),
             (0.491, 0.920),(-0.892, -0.946),(-0.721, -0.710),(0.519, -0.715),
              (-0.775, 0.551),(-0.646, 0.773),(-0.803, 0.878),(0.944, 0.801),
              (0.724, -0.795),(-0.748, -0.853),(-0.635, -0.905)])
Y = np.array([1]*9+[-1]*11)

#作图
x1 = X[Y>0][:,0]
y1 = X[Y>0][:,1]
x2 = X[Y<0][:,0]
y2 = X[Y<0][:,1]

plt.scatter(x1,y1)
plt.scatter(x2,y2)
plt.show()

#二次转换
clf = svm.SVC(kernel='poly',degree=2,coef0=1,gamma=1,C=1e10)
clf.fit(X,Y)

def g(x):
    r = np.sqrt(2)
    return np.array([1,r*x[0],r*x[1],x[0]**2,x[0]*x[1],x[1]*x[0],x[1]**2])

support = clf.support_
#y_n*a_n
x = np.array([g(i) for i in X])
coef = clf.dual_coef_[0]

#取第一个支持向量
s = support[0]

#获得系数
b = Y[s] - coef.dot(x[support].dot(x[s]))
k = (coef).dot(x[support])

#构造等高线函数
def g(x,y,k,b):
    r = np.sqrt(2)
    return k[0]+k[1]*r*x+k[2]*r*y+k[3]*(x**2)+(k[4]+k[5])*x*y+k[6]*(y**2)+b

#点的数量
n = 1000
r = 1

#作点
p = np.linspace(-r,r,n)
q = np.linspace(-r,r,n)

#构造网格
P,Q = np.meshgrid(p,q)

#绘制等高线
plt.contour(P,Q,g(P,Q,k,b),0)
plt.scatter(x1,y1)
plt.scatter(x2,y2)
plt.title('$\Phi_2$')
plt.show()


#三次转换
clf = svm.SVC(kernel='poly',degree=3,coef0=1,gamma=1,C=1e10)
clf.fit(X,Y)

def g2(x):
    r = np.sqrt(3)
    return np.array([1,r*x[0],r*x[1],r*x[0]**2,r*x[0]*x[1],r*x[1]*x[0],r*x[1]**2,
                     x[0]**3,r*x[0]**2*x[1],r*x[0]*x[1]**2,x[1]**3])

support = clf.support_
x = np.array([g2(i) for i in X])
coef = clf.dual_coef_[0]

#取第一个支持向量
s = support[0]

b = Y[s] - coef.dot(x[support].dot(x[s]))
k = (coef).dot(x[support])

#构造等高线函数
def g(x,y,k,b):
    r = np.sqrt(3)
    return k[0]+k[1]*r*x+k[2]*r*y+k[3]*r*x**2+k[4]*r*x*y+k[5]*r*y*x+k[6]*r*y**2+\
                     k[7]*x**3+k[8]*r*x**2*y+k[9]*r*x*y**2+k[10]*y**3+b

#点的数量
n = 1000
r = 1

#作点
p = np.linspace(-r,r,n)
q = np.linspace(-r,r,n)

#构造网格
P,Q = np.meshgrid(p,q)

#绘制等高线
plt.contour(P,Q,g(P,Q,k,b),0)
plt.scatter(x1,y1)
plt.scatter(x2,y2)
plt.title('$\Phi_3$')
plt.show()

####c
#转换数据
def h(x):
    x1 = x[0]
    x2 = x[1]
    return np.array([1,x1,x2,x1*x2,x1**2,x2**2,x1**3,(x1**2)*x2,x1*(x2**2),x2**3])

x3 = np.array([h(i) for i in X])

k = 1
size = x3.shape[1]
w = inv(x3.T.dot(x3)+k*np.eye(size)).dot(x3.T.dot(Y))

#作图
def s(x,y,w):
    return w[0]+w[1]*x+w[2]*y+w[3]*x*y+w[4]*(x**2)+w[5]*(y**2)+w[6]*(x**3)+w[7]*(x**2)*y+w[8]*x*(y**2)+w[9]*y**3

#点的数量
n = 1000
r = 1

#作点
p = np.linspace(-r,r,n)
q = np.linspace(-r,r,n)

#构造网格
P,Q = np.meshgrid(p,q)

#绘制等高线
plt.contour(P,Q,s(P,Q,w),0)

plt.scatter(x1,y1)
plt.scatter(x2,y2)
plt.show()