# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 14:04:41 2018

@author: Administrator
"""

import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

d=3
N=range(d+15,d+116,10)

def process(n,d=3,sigma=0.5,k=0.05):
    x=np.random.normal(size=(n,d))
    a=np.ones(n).reshape(n,1)
    X=np.concatenate((a,x),axis=1)
    k1=k/n
    w=np.random.normal(size=d+1)
    epsilon=np.random.normal(size=n)
    y=X.dot(w)+sigma*epsilon
    e1=0
    e2=0
    E=np.array([])
    for i in range(n):
        X1=np.concatenate((X[:i,:],X[i+1:,:]))
        y1=np.append(y[:i],y[i+1:])
        w=inv(X1.T.dot(X1)+k1*np.eye(d+1)).dot(X1.T.dot(y1))
        e=(X[i].dot(w)-y[i])**2
        E=np.append(E,e)
        if(i==1):
            e1=e
        elif i==2:
            e2=e
    return e1,e2,np.mean(E)

#(a)
#记录每个N对应的e1,e2,ecv
E={}
for n in N:
    E[n]=[]

for n in N:
    E1=np.array([])
    E2=np.array([])
    Ecv=np.array([])
    for i in range(10000):
        e1,e2,ecv=process(n,k=2.5)
        E1=np.append(E1,e1)
        E2=np.append(E2,e2)
        Ecv=np.append(Ecv,ecv)
    mean=(E1.mean(),E2.mean(),Ecv.mean())
    var=(E1.var(),E2.var(),Ecv.var())
    E[n].append(mean)
    E[n].append(var)

#(b)
E1=np.array([])
E2=np.array([])
Ecv=np.array([])
for n in N:
    temp=E[n][0]
    E1=np.append(E1,temp[0])
    E2=np.append(E2,temp[1])
    Ecv=np.append(Ecv,temp[2])
plt.plot(N,E1,label='e1')
plt.plot(N,E2,label='e2')
plt.plot(N,Ecv,label='ecv')
plt.title('mean')
plt.xlabel('N')
plt.legend()
plt.show()

#(e)
varE1=np.array([])
varE2=np.array([])
varEcv=np.array([])
for n in N:
    temp=E[n][1]
    varE1=np.append(varE1,temp[0])
    varE2=np.append(varE2,temp[1])
    varEcv=np.append(varEcv,temp[2])
plt.plot(N,N,label='$N$')
plt.plot(N,varE1/varEcv,label='$N_{eff}$')
plt.legend()
plt.show()


        