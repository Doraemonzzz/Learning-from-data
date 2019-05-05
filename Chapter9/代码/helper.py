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
    
def draw(X, y, model, n=500, flag=1, preprocess=None):
    """
    作图函数, flag=1表示不使用特征转换，flag=2表示使用whiten，其余情形使用PCA
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
    elif flag==2:
        data = preprocess.transform(np.c_[a.reshape(-1, 1), b.reshape(-1, 1)])
        aa, bb = np.reshape(data[:, 0], a.shape), np.reshape(data[:, 1], b.shape)
        #计算输出
        c = predict(aa, bb, model)
    else:
        #中心化
        data = preprocess.transform(np.c_[a.reshape(-1, 1), b.reshape(-1, 1)])
        #计算输出
        c = model.predict(data).reshape(a.shape)
        
    #作图
    plt.pcolormesh(a, b, c, cmap=cmap_light)
    plt.scatter(X[:, 0], X[:, 1], edgecolor='k', c=y, cmap=cmap_bold)
    plt.title("2-Class classification (k = %i)" % (model.n_neighbors))
    plt.show()

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
    
class StandardScaler_:
    def __init__(self):
        self.mean = None
        self.var = None
    
    def fit(self, X):
        N = X.shape[0]
        #中心化
        self.mean = np.mean(X, axis=0)
        X1 = X - self.mean
        #协方差
        Sigma = 1 / N * X1.T.dot(X1)
        #奇异值分解
        U, S, V = np.linalg.svd(Sigma)
        self.var = U.dot(np.diag(1 / np.sqrt(S))).dot(U.T)
    
    def fit_transform(self, X):
        self.fit(X)
        X1 = X - self.mean
        
        return X1.dot(self.var)
    
    def transform(self, X):
        X1 = X - self.mean
        
        return X1.dot(self.var)
    
class PCA_:
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean_ = None
        self.explained_variance_ratio_ = None
        self.U = None
        self.singular_values_ = None
        self.components_ = None
        self.flag = None
        
    def fit(self, X, flag=1):
        """
        flag=1表示中心化，否则不中心化
        """
        if flag == 1:
            #中心化
            self.mean_ = np.mean(X, axis=0)
            X -= self.mean_
            self.flag = flag

        #SVD分解
        U, S, V = np.linalg.svd(X, full_matrices=False)
        #np.linalg.svd返回的是V^T
        V = V.T
        self.U = U
        self.singular_values_ = S
        self.components_ = V
        #方差占比
        self.explained_variance_ratio_ = self.singular_values_ ** 2 / np.sum(self.singular_values_ ** 2)
        
        return self
    
    def transform(self, X):
        if self.flag == 1:
            X -= self.mean_
        Z = X.dot(self.components_[:, :self.n_components])
        
        return Z
    
    def fit_transform(self, X, flag=1):
        """
        flag=1表示中心化，否则不中心化
        """
        self.fit(X, flag)
        Z = self.transform(X)
        
        return Z
        