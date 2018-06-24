# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 08:18:34 2018

@author: Administrator
"""

import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers

#参数
rad=10
thk=5
sep=5

#n为产生点的个数,x1,y1为上半个圆环的坐标
def generatedata(rad,thk,sep,n,x1=0,y1=0):
    #上半个圆的圆心
    X1=x1
    Y1=y1

    #下半个圆的圆心
    X2=X1+rad+thk/2
    Y2=Y1-sep
    
    #上半个圆环的点
    top=[]
    #下半个圆环的点
    bottom=[]
    
    #后面要用到的参数
    r1=rad+thk
    r2=rad
    
    cnt=1
    while(cnt<=n):
        #产生均匀分布的点
        x=np.random.uniform(-r1,r1)
        y=np.random.uniform(-r1,r1)
        
        d=x**2+y**2
        if(d>=r2**2 and d<=r1**2):
            if (y>0):
                top.append([X1+x,Y1+y])
                cnt+=1
            else:
                bottom.append([X2+x,Y2+y])
                cnt+=1
        else:
            continue

    return top,bottom

#作图
n=1000
top,bottom=generatedata(rad,thk,sep,n)

X1=[i[0] for i in top]
Y1=[i[1] for i in top]

X2=[i[0] for i in bottom]
Y2=[i[1] for i in bottom]

plt.scatter(X1,Y1,s=1)
plt.scatter(X2,Y2,s=1)
plt.show()


#特征转换之前，算法1，sep=5

#加上偏移项
x1=[[1]+i for i in top]
x2=[[1]+i for i in bottom]

c=np.array([1.0,1.0,1.0])
A=np.concatenate((np.array(x1),np.array(x2)*(-1)))*(-1)
b=np.ones(n)*(-1.0)

c=matrix(c)
A=matrix(A)
b=matrix(b)

sol = solvers.lp(c,A,b)

w=list((sol['x']))

#作图
r=2*(rad+thk)
X3=[-r,r]
Y3=[-(w[0]+w[1]*i)/w[2] for i in X3]

plt.scatter(X1,Y1,s=1)
plt.scatter(X2,Y2,s=1)
plt.plot(X3,Y3)
plt.title('sep=5,algorithm1')
plt.show()
print(w)

#特征转换之前，算法2，sep=5
#根据之前所述构造矩阵
a1=np.concatenate((A,(-1)*np.eye(n)),axis=1)
a2=np.concatenate((np.zeros([n,3]),(-1)*np.eye(n)),axis=1)

A1=np.concatenate((a1,a2))
c1=np.array([0.0]*3+[1.0]*1000)
b1=np.array([-1.0]*1000+[0.0]*1000)

#带入算法求解
c1=matrix(c1)
A1=matrix(A1)
b1=matrix(b1)

sol1 = solvers.lp(c1,A1,b1)

#作图
w=list((sol1['x']))
#作图
r=2*(rad+thk)
X3=[-r,r]
Y3=[-(w[0]+w[1]*i)/w[2] for i in X3]

plt.scatter(X1,Y1,s=1)
plt.scatter(X2,Y2,s=1)
plt.plot(X3,Y3)
plt.title('sep=5,algorithm2')
plt.show()

#特征转换之前，算法1，sep=-5
sep=-5
top,bottom=generatedata(rad,thk,sep,n)

X1=[i[0] for i in top]
Y1=[i[1] for i in top]

X2=[i[0] for i in bottom]
Y2=[i[1] for i in bottom]

#加上偏移项
x1=[[1]+i for i in top]
x2=[[1]+i for i in bottom]

c=np.array([1.0,1.0,1.0])
A=np.concatenate((np.array(x1),np.array(x2)*(-1)))*(-1)
b=np.ones(n)*(-1.0)

c=matrix(c)
A=matrix(A)
b=matrix(b)

sol = solvers.lp(c,A,b)

#特征转换之前，算法2，sep=-5
#根据之前所述构造矩阵
a1=np.concatenate((A,(-1)*np.eye(n)),axis=1)
a2=np.concatenate((np.zeros([n,3]),(-1)*np.eye(n)),axis=1)

A1=np.concatenate((a1,a2))
c1=np.array([0.0]*3+[1.0]*1000)
b1=np.array([-1.0]*1000+[0.0]*1000)

#带入算法求解
c1=matrix(c1)
A1=matrix(A1)
b1=matrix(b1)

sol1 = solvers.lp(c1,A1,b1)

#作图
w=list((sol1['x']))
#作图
r=2*(rad+thk)
X3=[-r,r]
Y3=[-(w[0]+w[1]*i)/w[2] for i in X3]

plt.scatter(X1,Y1,s=1)
plt.scatter(X2,Y2,s=1)
plt.plot(X3,Y3)
plt.title('sep=-5,algorithm2')
plt.show()

#特征转换后，算法1，sep=-5
#data形式为[  1.        ,   3.05543009,  -3.72519952,  -1.        ]
#特征转换
def transform(data):
    result=[]
    for i in data:
        x1=i[1]
        x2=i[2]
        flag=i[-1]
        x=np.array([1,x1,x2,x1*x2,x1**2,x2**2,x1**3,(x1**2)*x2,x1*(x2**2),x2**3,flag])
        result.append(x)
    return result

#对数据预处理，加上标签和偏移项1
x1=[[1]+i+[1] for i in top]
x2=[[1]+i+[-1] for i in bottom]
data=x1+x2
newdata=transform(data)

finaldata=[]
for i in newdata:
    temp=list(i[:-1]*i[-1])
    finaldata.append(temp)
    
c=np.array([1.0]*10)
A=np.array(finaldata)*(-1)
b=np.ones(n)*(-1.0)

c=matrix(c)
A=matrix(A)
b=matrix(b)

sol = solvers.lp(c,A,b)

# 定义等高线高度函数
def f(x,y,w):
    return w[0]+w[1]*x+w[2]*y+w[3]*x*y+w[4]*(x**2)+w[5]*(y**2)+w[6]*(x**3)+w[7]*(x**2)*y+w[8]*x*(y**2)+w[9]*y**3

# 数据数目
m = 2000
#定义范围
t=25
# 定义x, y
x = np.linspace(-t, t, m)
y = np.linspace(-t, t, m)

# 生成网格数据
X, Y = np.meshgrid(x, y)


w=list((sol['x']))


plt.scatter(X1,Y1,s=1)
plt.scatter(X2,Y2,s=1)
plt.contour(X, Y, f(X, Y,w), 1, colors = 'red')
plt.title('featuretransform,sep=-5,algorithm1')
plt.show()
print(w)

#特征转换后，算法2，sep=-5
#根据之前所述构造矩阵
a1=np.concatenate((A,(-1)*np.eye(n)),axis=1)
a2=np.concatenate((np.zeros([n,10]),(-1)*np.eye(n)),axis=1)

A1=np.concatenate((a1,a2))
c1=np.array([0.0]*10+[1.0]*1000)
b1=np.array([-1.0]*1000+[0.0]*1000)

#带入算法求解
c1=matrix(c1)
A1=matrix(A1)
b1=matrix(b1)

sol1 = solvers.lp(c1,A1,b1)

#作图
w=list((sol1['x']))

# 定义x, y
x = np.linspace(-t, t, m)
y = np.linspace(-t, t, m)

# 生成网格数据
X, Y = np.meshgrid(x, y)


plt.scatter(X1,Y1,s=1)
plt.scatter(X2,Y2,s=1)
plt.contour(X, Y, f(X, Y,w), 1, colors = 'red')
plt.title('featuretransform,sep=-5,algorithm2')
plt.show()