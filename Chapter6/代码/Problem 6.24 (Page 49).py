# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 11:27:42 2019

@author: qinzhen
"""

import numpy as np
import matplotlib.pyplot as plt
import helper as hlp

#### (a)
n = 1000
X, y = hlp.Data(n, scale=0.5) 
plt.scatter(X[:, 0], X[:, 1], edgecolor='k', c=y)
plt.show()

n = X.shape[0]
K = np.arange(1, 10)
Ein = []
for k in K:
    #训练模型
    kmeans = hlp.KMeans_(k)
    kmeans.fit(X)
    #获得标签
    label = kmeans.labels_
    #获得聚类中心
    center = kmeans.cluster_centers_
    #计算Ein
    ein = 0
    for i in range(k):
        #计算每一类的误差
        ein += np.sum((X[label==i] - center[i]) ** 2)
    #计算均值
    ein /= n
    Ein.append(ein)

#作图
plt.plot(K, Ein)
plt.title("$K$ VS $E_{in}$")
plt.xlabel("$K$")
plt.ylabel("$E_{in}$")
plt.show()

#### (b)
#记录结果
Ein_rand = []
#试验次数
N = 1000

for k in K:
    ein_k = []
    for _ in range(N):
        X, y = hlp.Data(n, scale=0.5) 
        #计算范围
        X1_min = np.min(X[:, 0])
        X1_max = np.max(X[:, 0])
        X2_min = np.min(X[:, 1])
        X2_max = np.max(X[:, 1])
        #生成点
        X1_rand = np.random.uniform(X1_min, X1_max, size=n)
        X2_rand = np.random.uniform(X2_min, X2_max, size=n)
        #合并
        Xrand = np.c_[X1_rand, X2_rand]
        #训练模型
        kmeans = hlp.KMeans_(k)
        kmeans.fit(Xrand)
        #获得标签
        label = kmeans.labels_
        #获得聚类中心
        center = kmeans.cluster_centers_
        #计算Ein
        ein = 0
        for i in range(k):
            #计算每一类的误差
            ein += np.sum((Xrand[label==i] - center[i]) ** 2)
        #计算均值
        ein /= n
        #存储结果
        ein_k.append(ein)
    Ein_rand.append(np.mean(ein_k))
    
plt.plot(K, Ein_rand)
plt.title("$K$ VS $E^{{rand}}_{{in}} (k)$")
plt.xlabel("$K$")
plt.ylabel("$E^{{rand}}_{{in}} (k)$")
plt.show()

#### (c)
#计算Gk
Gk = np.log(Ein_rand) - np.log(Ein)

#作图
plt.plot(K, Gk)
plt.title("$K$ VS $G(k)$")
plt.xlabel("$K$")
plt.ylabel("$G(k)$")
plt.show()

#最优解
k_opt = K[np.argmin(np.diff(Gk)) + 1]
print("Optimal choice of k is {}".format(k_opt))

#### (d)
#试验次数
N = 100
Sigma = [0.1, 0.3, 0.5, 0.75, 1]
K_opt = []

for sigma in Sigma:
    #记录结果
    Ein_rand_sigma = []
    for k in K:
        ein_k = []
        for _ in range(N):
            X, y = hlp.Data(n, scale=sigma) 
            #计算范围
            X1_min = np.min(X[:, 0])
            X1_max = np.max(X[:, 0])
            X2_min = np.min(X[:, 1])
            X2_max = np.max(X[:, 1])
            #生成点
            X1_rand = np.random.uniform(X1_min, X1_max, size=n)
            X2_rand = np.random.uniform(X2_min, X2_max, size=n)
            #合并
            Xrand = np.c_[X1_rand, X2_rand]
            #训练模型
            kmeans = hlp.KMeans_(k)
            kmeans.fit(Xrand)
            #获得标签
            label = kmeans.labels_
            #获得聚类中心
            center = kmeans.cluster_centers_
            #计算Ein
            ein = 0
            for i in range(k):
                #计算每一类的误差
                ein += np.sum((Xrand[label==i] - center[i]) ** 2)
            #计算均值
            ein /= n
            #存储结果
            ein_k.append(ein)
        Ein_rand_sigma.append(np.mean(ein_k))
    Gk = np.log(Ein_rand_sigma) - np.log(Ein)
    #记录最优k
    k_opt = K[np.argmin(np.diff(Gk)) + 1]
    K_opt.append(k_opt)

#作图
plt.plot(Sigma, K_opt)
plt.title("$\sigma$ VS $k$")
plt.xlabel("$\sigma$")
plt.ylabel("$k$")
plt.show()