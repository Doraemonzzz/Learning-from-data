# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 09:47:05 2018

@author: Administrator
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from sklearn import linear_model


def process(l,Qf=5,N=100,sigma2=1,d=2):
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
    
    #系数ai
    a=np.random.normal(size=Qf+1)
    
    #标准化
    k=np.arange(1,2*Qf+2,2)
    s=(2*a**2/k).sum()
    a=a/np.sqrt(s)
    
    #产生点集
    x=np.random.uniform(low=-1,high=1,size=N)
    x.sort()
    #计算之前所述的X
    X=[]
    for i in x:
        temp=L(Qf,i)
        X.append(temp)
    X=np.array(X)
    #差生误差项
    epsilon=np.sqrt(sigma2)*np.random.normal(size=N)
    #计算Y
    Y=X.dot(a.T)+epsilon
    
    #### Step 2:拟合数据
    
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
    X0=tranform(x,d)
    
    #计算Lasso回归,Ridge回归
    #Lasso
    lasso=linear_model.Lasso(alpha=l)
    lasso.fit(X0,Y)
    #print(lasso.coef_)
    
    #Ridge
    ridge=linear_model.Ridge(alpha=l)
    ridge.fit(X0,Y)
    
    #print(ridge.coef_)
    #r=np.linalg.inv((X0.T.dot(X0)+l*np.eye(d+1))).dot(X0.T.dot(Y))
    
    #### Step 3:计算结果
    
    #构造被积函数,a为系数
    def E(x,w,a):
        #计算f(x)
        #n为勒让德多项式次数
        n=len(a)-1
        l=L(n,x)
        f=a.dot(l)
        
        #计算g(x)
        X=np.array([x**i for i in range(len(w))])
        g=X.dot(w)
        
        return (g-f)**2/2
    
    El=quad(E, -1, 1, args=(lasso.coef_,a))[0]
    Eq=quad(E, -1, 1, args=(ridge.coef_,a))[0]
    #Eq=quad(E, -1, 1, args=(r,a))[0]

    return El,Eq,lasso.coef_,ridge.coef_

L=np.arange(0.01,2.01,0.05)
El=[]
Eq=[]
lasso=[]
ridge=[]
for l in L:
    temp=process(l)
    El.append(temp[0])
    Eq.append(temp[1])
    
plt.plot(L,El,label='Lasso')
plt.plot(L,Eq,label='Ridge')
plt.xlabel('$\lambda$')
plt.title('$E_{out}$')
plt.legend()
plt.show()


lasso=[]
ridge=[]
for l in L:
    temp=process(l,Qf=20,N=3,d=5)
    lasso.append(np.sum(temp[2]!=0))
    ridge.append(np.sum(temp[3]!=0))
plt.plot(L,lasso,label='lasso')
plt.plot(L,ridge,label='ridge')
plt.title('number of non-zero weights')
plt.legend()
plt.xlabel('$\lambda$')
plt.show()