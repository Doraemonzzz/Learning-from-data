# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 12:47:12 2019

@author: qinzhen
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

def Data(n, flag=1, scale=0.15):
    """
    按照Problem 13生成n个点，flag=1表示圆心固定
    """
    if flag:
        #中心
        center = np.array([[0, 1, 0, -1], [1, 0, -1, 0]]).T
    else:
        #按极坐标方式生成点
        theta = np.random.uniform(0, 2 * np.pi, size=4)
        center = np.c_[np.cos(theta), np.sin(theta)]
    #生成X
    X = np.random.normal(scale=scale, size=(n, 2))
    #生成每个数据对应的分类
    index = np.random.randint(0, 4, size=n)
    #增加中心
    X += center[index] 
    #生成标签
    y = np.copy(index)
    y[y%2==1] = -1
    y[y%2==0] = 1
    
    return X, y

def predict(a, b, model):
    """
    预测每个网格点的输出
    输入: a, b为两个m*n的矩阵, model为模型
    输出: 每个(a[i][j], b[i][j])的输出构成的矩阵
    """
    #将网格拉直并拼接
    X = np.c_[a.reshape(-1, 1), b.reshape(-1, 1)]
    #预测
    label = model.predict(X)
    #恢复成网格形状
    label = np.reshape(label, np.shape(a))
    
    return label

class KNeighborsClassifier_():
    def __init__(self, n_neighbors):
        self.n_neighbors = n_neighbors
        
        
    def fit(self, X, y):
        self.X = X
        self.y = y
        
    def predict(self, X):
        #计算距离矩阵
        d1 = np.sum(X ** 2, axis=1).reshape(-1, 1)
        d2 = np.sum(self.X ** 2, axis=1).reshape(1, -1)
        dist = d1 + d2 - 2 * X.dot(self.X.T)
        
        #找到最近的k个点的索引
        index = np.argsort(dist, axis=1)[:, :self.n_neighbors]
        #计算预测结果
        y = np.sign(np.sum(self.y[index], axis=1))
        
        return y
    
def draw(X, y, model, n=500, flag=1):
    """
    作图函数, flag=1表示不使用特征转换，其余情况使用特征转换
    """
    # Create color maps
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00'])
    
    x1_min, x1_max = X[:, 0].min() - 2, X[:, 0].max() + 2
    x2_min, x2_max = X[:, 1].min() - 2, X[:, 1].max() + 2
    
    #生成网格数据
    a, b = np.meshgrid(np.linspace(x1_min, x1_max, n),
                       np.linspace(x2_min, x2_max, n))
    if flag==1:
        #计算输出
        c = predict(a, b, model)
    else:
        aa = np.sqrt(a * a + b * b)
        bb = np.arctan(b / (a + 10**(-8)))
        #计算输出
        c = predict(aa, bb, model)
        
    #作图
    plt.pcolormesh(a, b, c, cmap=cmap_light)
    plt.scatter(X[:, 0], X[:, 1], edgecolor='k', c=y, cmap=cmap_bold)
    plt.title("2-Class classification (k = %i)" % (model.n_neighbors))
    plt.show()
    
def generatedata(rad, thk, sep, n, x1=0, y1=0):
    """
    产生课本109页的数据集，这里设置的参数为半径rad，圆环宽度thk，
    上下圆环间隔sep，n为数据集总数，x1, y1为上半圆环的圆心
    """
    #上半个圆的圆心
    X1 = x1
    Y1 = y1

    #下半个圆的圆心
    X2 = X1 + rad + thk / 2
    Y2 = Y1 - sep
    
    #生成角度theta
    Theta = np.random.uniform(0, 2*np.pi, n)
    #生成距离r
    R = np.random.uniform(rad, rad+thk, n)
    
    #根据Theta生成标签
    y = 2 * (Theta < np.pi) - 1
    
    #生成点集合X，首先根据y的标签生成圆心
    X = np.zeros((n, 2))
    X[y > 0] = np.array([X1, Y1]) 
    X[y < 0] = np.array([X2, Y2])
    #其次用参数方程生成坐标
    X[:, 0] += np.cos(Theta) * R
    X[:, 1] += np.sin(Theta) * R
    
    return X, y

def CNN(X, y, k=1):
    n = X.shape[0]
    #初始化
    index = np.random.choice(np.arange(n), size=k, replace=False)
    #condense data
    X_cd = X[index, :]
    y_cd = y[index]
    #剩余的点
    X1 = np.delete(X, index, axis=0)
    y1 = np.delete(y, index, axis=0)
    
    Ein = []
    while True:
        #训练knn
        nn = KNeighborsClassifier_(1)
        nn.fit(X_cd, y_cd)
        #预测结果
        y_pred = nn.predict(X)
        #错误率
        ein = np.mean(y_pred != y)
        Ein.append(ein)
        
        if ein != 0:
            #找到分类错误的点
            i1 = np.where(y_pred != y)[0][0]
            #找到y1中和y[ii]相同的点的索引
            i2 = np.where(y1 == y[i1])[0][0]
            #添加至condese data
            X_cd = np.r_[X_cd, X1[i2, :].reshape(1, 2)]
            y_cd = np.r_[y_cd, y1[i2]]
            #删除该数据
            X1 = np.delete(X1, i2, axis=0)
            y1 = np.delete(y1, i2, axis=0)
        else:
            break
        
    return X_cd, y_cd

class KMeans_():
    def __init__(self, k, D=1e-5):
        #聚类数量
        self.k = k
        #聚类中心
        self.cluster_centers_ = []
        #聚类结果
        self.labels_ = []
        #设置阈值
        self.D = D
        
    def fit(self, X):
        #数据维度
        n, d = X.shape
        #聚类标签
        labels = np.zeros(n, dtype=int)
        #初始中心点
        index = np.random.randint(0, n, self.k)
        cluster_centers = X[index]
        #记录上一轮迭代的聚类中心
        cluster_centers_pre = np.copy(cluster_centers)
        
        while True:
            #计算距离矩阵
            d1 = np.sum(X ** 2, axis=1).reshape(-1, 1)
            d2 = np.sum(cluster_centers ** 2, axis=1).reshape(1, -1)
            dist = d1 + d2 - 2 * X.dot(cluster_centers.T)
            
            #STEP1:找到最近的中心
            labels = np.argmin(dist, axis=1)
            #STEP2:重新计算中心
            for i in range(self.k):
                #第i类的索引
                index = (labels==i)
                #第i类的数据
                x = X[index]
                #判断是否有点和某聚类中心在一类
                if len(x) != 0:
                    cluster_centers[i] = np.mean(x, axis=0)
                
            #计算误差
            delta = np.linalg.norm(cluster_centers - cluster_centers_pre)
            
            if delta < self.D:
                break
            
            cluster_centers_pre = np.copy(cluster_centers)
            
        self.cluster_centers_ = np.copy(cluster_centers)
        self.labels_ = labels
        
        
    def predict(self, X):
        #计算距离矩阵
        d1 = np.sum(X ** 2, axis=1).reshape(-1, 1)
        d2 = np.sum(self.cluster_centers_ ** 2, axis=1).reshape(1, -1)
        dist = d1 + d2 - 2 * X.dot(self.cluster_centers_.T)
        
        #找到最近的中心
        self.cluster_centers_ = np.argmin(dist, axis=1)
        
        return self.cluster_centers_