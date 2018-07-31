# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 01:14:35 2018

@author: Administrator
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.linear_model import Perceptron

####(a)
####作图
N=20
x1=np.random.uniform(0,1,N)
x2=np.random.uniform(-1,1,N)
x=[]
for i in range(N):
    x.append([x1[i],x2[i]])
x=np.array(x)
y=np.sign(x2)

plt.scatter(x1[y>0],x2[y>0],label="+")
plt.scatter(x1[y<0],x2[y<0],label="-")
plt.legend()
plt.show()

####训练数据
clf=svm.SVC(kernel ='linear',C=1e10)
clf.fit(x,y)

#获得超平面
w=clf.coef_[0]
b=clf.intercept_[0]

#作图
xx=np.array([-1,1])
yy=-(b+w[0]*xx)/w[1]
plt.scatter(x1[y>0],x2[y>0],label="+")
plt.scatter(x1[y<0],x2[y<0],label="-")
plt.plot(xx,yy,'r')
plt.legend()
plt.show()

#计算margin
margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
print("margin =",margin)

#计算Eout
def Eout(w,b,N=1000):
    x1=np.random.uniform(0,1,N)
    x2=np.random.uniform(-1,1,N)
    x=[]
    for i in range(N):
        x.append([x1[i],x2[i]])
    x=np.array(x)
    y=np.sign(x2)
    y1=np.sign(x.dot(w)+b)
    return np.sum(y!=y1)/N
e=Eout(w,b)
print(e)

####(c)
clf=Perceptron()
clf.fit(x,y)

w1=clf.coef_[0]
b1=clf.intercept_[0]

#作图
xx=np.array([-1,1])
yy=-(b1+w1[0]*xx)/w1[1]
plt.scatter(x1[y>0],x2[y>0],label="+")
plt.scatter(x1[y<0],x2[y<0],label="-")
plt.plot(xx,yy,'r')
plt.legend()
plt.show()

#多次实验，做直方图
result=np.array([])
for i in range(2000):
    np.random.shuffle(x)
    y=np.sign(x[:,1])
    clf.fit(x,y)
    w1=clf.coef_[0]
    b1=clf.intercept_[0]
    result=np.append(result,Eout(w1,b1))
    
plt.hist(result)
plt.plot([e]*400,range(400),'r')
plt.show()