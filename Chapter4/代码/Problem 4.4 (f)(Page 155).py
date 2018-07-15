# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 10:55:56 2018

@author: Administrator
"""

import numpy as np
from cvxopt import matrix, solvers
import matplotlib.pyplot as plt
from scipy.integrate import quad
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

#参数
Qf=20
N=11
sigma2=1

#### Step 1:数据准备

#定义勒让德多项式，产生L(0,x),...,L(k,x),注意这里不要用递归
def L(k,x):
    if(k==0):
        return [1.0]
    elif(k==1):
        return [1.0,x*1.0]
    else:
        result=[1,x]
        for i in range(2,k+1):
            s=(2*i-1)/i*(x*result[-1])-(i-1)/i*result[-2]
            result.append(s)
        return result

#系数ai，首项为0
a=np.random.normal(size=Qf)
#a=np.append(a,)

#标准化
k=np.arange(3,2*Qf+2,2)
s=(2*a**2/k).sum()
a=a/np.sqrt(s)

#产生点集
x=np.random.uniform(low=-1,high=1,size=N)
x.sort()
#计算之前所述的X
X=[]
for i in x:
    temp=L(Qf,i)[1:]#注意这里从第二项取，因为第一项系数为0
    X.append(temp)
X=np.array(X)
#差生误差项
epsilon=np.sqrt(sigma2)*np.random.normal(size=N)
#计算Y
Y1=X.dot(a.T)+epsilon
Y=np.sign([Y1]).T


#### Step 2:利用Problem 3.6,3.7的方法求最优解

#对一个数据特征转换,将x转换为(1,x,...,x^k)
def t(x,k):
    result=[x**i for i in range(k+1)]
    return result

#对一组数据x=[x1,...xN]做特征转换
def tranform(X,k):
    result=[]
    for x in X:
        temp=t(x,k)
        result.append(temp)
    return np.array(result)

#特征转换
X2=tranform(x,2)
X10=tranform(x,10)

def LP(X,N):
    a1=np.concatenate((-Y*X,(-1)*np.eye(N)),axis=1)
    a2=np.concatenate((np.zeros([N,X.shape[1]]),(-1)*np.eye(N)),axis=1)
    A=np.concatenate((a1,a2))
    b=np.array([-1.0]*N+[0.0]*N)
    c=np.append(np.zeros(X.shape[1]),np.ones(N))
    #带入算法求解
    A=matrix(A)
    b=matrix(b)
    c=matrix(c)
    
    sol = solvers.lp(c,A,b)
    return sol

sol2=LP(X2,N)
sol10=LP(X10,N)
w2=list((sol2['x']))[:3]
w10=list((sol10['x']))[:11]

#### Step 3:计算误差
def E(w,a,N):
    X=np.random.uniform(low=-1,high=1,size=N)
    #n为勒让德多项式次数
    n=len(a)
    result=0
    for x in X:
        l=L(n,x)[1:]
        f=a.dot(l)+np.random.normal()
        
        #计算g(x)
        x1=np.array([x**i for i in range(len(w))])
        g=x1.dot(w)
        result+=np.sum(np.sign(g)!=np.sign(f))
    
    return result/N

print(E(w2,a,100))
print(E(w10,a,100))

#### Step 4:作图
def f(x, w, n):
    X=tranform(x,n)
    return np.array(w[:n+1]).dot(X.T)



x1=x[Y1>0]
y1=Y1[Y1>0]
x2=x[Y1<0]
y2=Y1[Y1<0]
x3=np.arange(-1,1,0.1)
y3=f(x3,w2,2)
y4=f(x3,w10,10)

plt.scatter(x1,y1,s=1)
plt.scatter(x2,y2,s=1)
plt.plot(x3,y3)
plt.plot(x3,y4)
plt.show()



