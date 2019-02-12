# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 13:02:55 2019

@author: qinzhen
"""
import numpy as np
import matplotlib.pyplot as plt
import helper as hlp
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

def Adaline(n1, n2, d, rnd, eta, w0, iteration=1000):
    """
    生成n1+n2个d维点（不包括偏置项1），n1个点作为训练数据，n2个点作为测试数据，
    x1+...+xd>=t的点标记为+1，x1+...+xd<=-t的点标记为-1，
    w0为分界线的法向量，w0=[0] + [1] * d，这里d = 2
    """
    #生成数据
    X_train, y_train = hlp.data(n1, d, rnd)
    X_test, y_test = hlp.data(n2, d, rnd)
    
    #记录次数
    T = 0
    w = np.zeros(d + 1)
    #print(X_train.dot(w) * y_train)
    while(hlp.Judge(X_train, y_train, w)==False and T < iteration):
        i = np.random.randint(0, n1)
        s = X_train[i, :].dot(w)
        a = s * y_train[i]
        if a <= 1:
            w += eta * (y_train[i] - s) * X_train[i, :]
            T += 1
    
    #计算错误率
    num = np.sum(X_test.dot(w) * y_test <= 0)
    
    print("n为"+str(eta)+"时错误率为"+str(num / n2))
    
    #直线方程为w0+w1*x+w2*y=0,根据此生成点
    X3 = np.arange(-1, 1, 0.1)
    Y3 = np.array([(X3[i]*w[1]+w[0])/(-w[2]) for i in range(len(X3))])

    #目标函数
    X4 = np.arange(-1, 1, 0.1)
    Y4 = np.array([(X3[i]*w0[1]+w0[0])/(-w0[2]) for i in range(len(X4))])

    #画出图片
    plt.scatter(X_train[y_train == 1][:, 1], X_train[y_train == 1][:, 2], c='r', s=1)
    plt.scatter(X_train[y_train == -1][:, 1], X_train[y_train == -1][:, 2], c='b', s=1)
    plt.plot(X3, Y3, label="("+str(w[0])+")+("+str(w[1])+")x+("+str(w[2])+")y=0")
    plt.plot(X4, Y4, label="("+str(w0[0])+")+("+str(w0[1])+")x+("+str(w0[2])+")y=0")
    plt.title(u"经过"+str(T)+u"次迭代")
    #设置坐标范围
    #plt.xticks(np.arange(0,10))
    #plt.yticks(np.arange(0,10))
    plt.legend()
    plt.show()

#设置随机种子，保证每次结果一致
seed = 42
rnd = np.random.RandomState(seed)  
Eta = [1, 0.1, 0.01, 0.001]
for eta in Eta:
    Adaline(100, 10000, 2, rnd, eta, [0, 1, 1])