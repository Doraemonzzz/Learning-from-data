# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 13:19:20 2019

@author: qinzhen
"""

import numpy as np
import matplotlib.pyplot as plt
import helper as hlp


#### (a)
# =============================================================================
# CNN1
# =============================================================================
def CNN1(X, y, k=1):
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
        nn = hlp.KNeighborsClassifier_(1)
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

# =============================================================================
# CNN2
# =============================================================================
def dist(X1, X2):
    """
    计算X1, X2每一项之间的距离
    """
    d1 = np.sum(X1 ** 2, axis=1).reshape(-1, 1)
    d2 = np.sum(X2 ** 2, axis=1).reshape(1, -1)
    dist = d1 + d2 - 2 * X1.dot(X2.T)
    
    return dist

def influence_set(d1, d2):
    """
    根据距离生成全体influence_set, 用字典存储
    d1为标签相同的点的距离, d2为标签不同的点的距离
    """
    #计算不同标签的最短距离
    d3 = np.min(d2, axis=1)
    #找到相同标签中小于最短距离的部分
    i1 = d1 < d3
    #找到influence set对应的索引
    u, v = np.where(i1)
    #生成influence set
    influe_set = {}
    for i in np.unique(u):
        #使用集合
        influe_set[i] = set(v[u==i])
        
    return influe_set

def influe_set_helper(i, key, X, influe_set):
    """
    删除元素最多的influence set并返回对应的x
    key为influence set所在字典的键，i为元素最多的influence set的索引，
    X为点集，influe_set为influence set所在字典
    """
    
    index = key[i]
    #需要删除的集合
    x_del = influe_set[index]
    x_del.add(index)
    #influence set所在的字典
    for j in key:
        #如果索引不是influence set对应的x，则更新其influence set
        if j != index:
            #删除公共元素
            influe_set[j] -= x_del
            #如果为空，则删除该influence set
            if influe_set[j] == set():
                influe_set.pop(j)
        #如果索引是influence set对应的x，则删除influence set
        else:
            influe_set.pop(j)
    
    return influe_set, X[index]

def CNN2(X, y):
    #将数据分为两类
    X_pos = X[y==1]
    X_neg = X[y==-1]
    
    #计算距离矩阵
    dist_pos = dist(X_pos, X_pos)
    
    #设置对角线的值
    np.fill_diagonal(dist_pos, np.inf)
    dist_neg = dist(X_neg, X_neg)
    
    #设置对角线的值
    np.fill_diagonal(dist_neg, np.inf)
    dist_pos2neg = dist(X_pos, X_neg)
    
    #influcence set
    influe_set_pos = influence_set(dist_pos, dist_pos2neg) 
    influe_set_neg = influence_set(dist_neg, dist_pos2neg.T)

    #condense data
    X_cd = []
    y_cd = []
    while True:
        #计算influence set非空的数量
        len_pos = np.array([len(influe_set_pos[i]) for i in influe_set_pos])
        len_neg = np.array([len(influe_set_neg[i]) for i in influe_set_neg])
        #得到键
        pos_key = [i for i in influe_set_pos]
        neg_key = [i for i in influe_set_neg]

        
        if (len(len_pos) > 0 and len(len_neg) > 0):
            #找到最多元素的influence set
            i1 = np.argmax(len_pos)
            i2 = np.argmax(len_neg)
            if len_pos[i1] > len_neg[i2]:
                influe_set_pos, x = influe_set_helper(i1, pos_key, X_pos, influe_set_pos)
                X_cd.append(x)
                y_cd.append(1)
            else:
                influe_set_neg, x = influe_set_helper(i2, neg_key, X_neg, influe_set_neg)
                X_cd.append(x)
                y_cd.append(-1)
        elif len(len_pos) > 0:
            i1 = np.argmax(len_pos)
            influe_set_pos, x = influe_set_helper(i1, pos_key, X_pos, influe_set_pos)
            X_cd.append(x)
            y_cd.append(1)
        elif len(len_neg) > 0:
            i2 = np.argmax(len_neg)
            influe_set_neg, x = influe_set_helper(i2, neg_key, X_neg, influe_set_neg)
            X_cd.append(x)
            y_cd.append(-1)
        else:
            break
        
    return np.array(X_cd), np.array(y_cd)


#### (b)
n = 1000
X, y = hlp.Data(n, 1, scale=0.5) 
plt.scatter(X[:, 0], X[:, 1], edgecolor='k', c=y)
plt.show()
#原始数据的分类结果
nn = hlp.KNeighborsClassifier_(1)
nn.fit(X, y)
hlp.draw(X, y, nn, n=500, flag=1)

#### CNN1
#condense data的分类结果
X_cd, y_cd = CNN1(X, y)
nn = hlp.KNeighborsClassifier_(1)
nn.fit(X_cd, y_cd)
hlp.draw(X_cd, y_cd, nn, n=500, flag=1)

#### CNN2
#condense data的分类结果
X_cd, y_cd = CNN2(X, y)
nn = hlp.KNeighborsClassifier_(1)
nn.fit(X_cd, y_cd)
hlp.draw(X_cd, y_cd, nn, n=500, flag=1)

'''
#### 运行时间较长，不建议运行
#### (c)
m = 1000
NUM1 = []
NUM2 = []
for i in range(m):
    X, y = hlp.Data(n, 1)
    X_cd1, y_cd1 = CNN1(X, y)
    NUM1.append(len(y_cd1))
    
    X_cd2, y_cd2 = CNN2(X, y)
    NUM2.append(len(y_cd2))
    
plt.hist(NUM1)
plt.title("CNN1")
plt.show()
print("Average sizes of the condensed sets is {}(CNN1)".format(np.mean(NUM1)))

plt.hist(NUM2)
plt.title("CNN2")
plt.show() 
print("Average sizes of the condensed sets is {}(CNN2)".format(np.mean(NUM2)))
'''