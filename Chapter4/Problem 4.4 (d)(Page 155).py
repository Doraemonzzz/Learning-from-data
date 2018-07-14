# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 12:34:01 2018

@author: Administrator
"""

import numpy as np
import pickle
from numpy.linalg import inv
from scipy.integrate import quad
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

def process(Qf,N,sigma2):
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
#    Y1=X.dot(a.T)
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
    X2=tranform(x,2)
    X10=tranform(x,10)
    
    #计算结果
    w2=inv(X2.T.dot(X2)).dot(X2.T).dot(Y)
    w10=inv(X10.T.dot(X10)).dot(X10.T).dot(Y)
    
    
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
    
    E2=quad(E, -1, 1, args=(w2,a))[0]
    E10=quad(E, -1, 1, args=(w10,a))[0]

    return E10-E2


#############################
##stochastic noise
    
'''
E={}
t=100

Qf=[20]
N=np.arange(5,122,5)
sigma2=np.arange(0,2.01,0.05)
q=20

for n in N:
    for s in sigma2:
        e=np.array([])
        i=0
        while (i<t):
            error=process(q,n,s)
            e=np.append(e,error)
            i+=1
        E[(q,n,s)]=np.mean(e)

#保存数据
#with open('stochastic noise.pickle','wb') as f:
#    pickle.dump(E,f)


#如果不训练读取数据即可
#读取数据
#pickle_in=open('stochastic noise.pickle','rb')
#E=pickle.load(pickle_in)

          
c=[]
q1=[]
n1=[]
for i in E:
    q1.append([i[1]])
    n1.append([i[2]])
    c.append([E[i]])

c=np.array(c)
q1=np.array(q1)
n1=np.array(n1)

n1,q1=np.meshgrid(n1,q1)
result=[]
for i in range(len(q1)):
    q=q1[i]
    n=n1[i]
    temp=[]
    for j in range(len(q)):
        temp.append(E[(20,q[j],n[j])])
    result.append(temp)

cm=plt.cm.get_cmap('rainbow')
plt.pcolormesh(q1,n1,result,cmap=cm,vmin=-0.2,vmax=0.2,shading='gouraud',edgecolors='face')
plt.xlabel("Number of Data Points, N")
plt.ylabel("Noise Level,sigma^2")
plt.title("stochastic noise")
plt.colorbar()
plt.show()
'''

#############################
##stochastic noise

'''
Qf=np.arange(1,31)
N=np.arange(5,121,5)
s=sigma2=0.1

E={}
t=150

for q in Qf:
    for n in N:
        e=np.array([])
        i=0
        while (i<t):
            error=process(q,n,s)
            e=np.append(e,error)
            i+=1
        E[(q,n,s)]=np.mean(e)



#保存数据
#with open('deterministic noise.pickle','wb') as f:
#    pickle.dump(E,f)


#如果不训练读取数据即可
#读取数据
#pickle_in=open('deterministic noise.pickle','rb')
#E=pickle.load(pickle_in)


c=[]
q1=[]
n1=[]
for i in E:
    q1.append([i[0]])
    n1.append([i[1]])
    c.append([E[i]])

c=np.array(c)
q1=np.array(q1)
n1=np.array(n1)

n1,q1=np.meshgrid(n1,q1)
result=[]
for i in range(len(q1)):
    q=q1[i]
    n=n1[i]
    temp=[]
    for j in range(len(q)):
        temp.append(E[(q[j],n[j],0.1)])
    result.append(temp)

        
cm=plt.cm.get_cmap('rainbow')
plt.pcolormesh(n1,q1,result,cmap=cm,vmin=-0.2,vmax=0.2,shading='gouraud',edgecolors='face')
plt.colorbar()
plt.xlabel("Number of Data Points, N")
plt.ylabel("Target Complexity, Qf")
plt.title("deterministic noise")
plt.show()
'''