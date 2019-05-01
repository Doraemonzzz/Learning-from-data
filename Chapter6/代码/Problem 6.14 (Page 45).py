# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 20:19:55 2019

@author: qinzhen
"""

import helper as hlp
import numpy as np

#### (a)
#获得数据
Train = np.genfromtxt("features.train")
Test = np.genfromtxt("features.test")

n = 500
m = 100
y_train, X_train = Train[: n, 0], Train[: n, 1:]
y_train[y_train != 1] = -1
y_test, X_test = Test[: m, 0], Test[: m, 1:]
y_test[y_test != 1] = -1

#### (b)
#训练模型
nn = hlp.KNeighborsClassifier_(3)
nn.fit(X_train, y_train)
y_test_pred = nn.predict(X_test)
y_train_pred = nn.predict(X_train)

#计算错误率
ein = np.mean(y_train != y_train_pred)
etest = np.mean(y_test != y_test_pred)
print("Ein of 3-NN is {}".format(ein))
print("Etest of 3-NN is {}".format(etest))
#作图
hlp.draw(X_train, y_train, nn, n=500, flag=1)

#### (c)
X_cd, y_cd = hlp.CNN(X_train, y_train, k=3)
#训练模型
nn = hlp.KNeighborsClassifier_(3)
nn.fit(X_cd, y_cd)
y_test_pred = nn.predict(X_test)
y_train_pred = nn.predict(X_train)

#计算错误率
ein = np.mean(y_train != y_train_pred)
etest = np.mean(y_test != y_test_pred)
print("Ein of C-NN is {}".format(ein))
print("Etest of C-NN is {}".format(etest))
#作图
hlp.draw(X_cd, y_cd, nn, n=500, flag=1)

#### (d)
N = 1000
#索引
train_index = np.arange(Train.shape[0])
test_index = np.arange(Test.shape[0])

Ein_KNN = []
Etest_KNN = []
Ein_CNN = []
Etest_CNN = []

for i in range(N):
    #训练数据
    train = Train[np.random.choice(train_index, size=n, replace=False)]
    y_train, X_train = train[:, 0], train[:, 1:]
    y_train[y_train != 1] = -1
    #测试数据
    test = Test[np.random.choice(test_index, size=m, replace=False)]
    y_test, X_test = test[:, 0], test[:, 1:]
    y_test[y_test != 1] = -1
    
    #训练KNN
    nn = hlp.KNeighborsClassifier_(3)
    nn.fit(X_train, y_train)
    y_test_pred = nn.predict(X_test)
    y_train_pred = nn.predict(X_train)

    #计算错误率
    ein = np.mean(y_train != y_train_pred)
    etest = np.mean(y_test != y_test_pred)
    Ein_KNN.append(ein)
    Etest_KNN.append(etest)
    
    #训练CNN
    X_cd, y_cd = hlp.CNN(X_train, y_train, k=3)
    #训练模型
    nn = hlp.KNeighborsClassifier_(3)
    nn.fit(X_cd, y_cd)
    y_test_pred = nn.predict(X_test)
    y_train_pred = nn.predict(X_train)

    #计算错误率
    ein = np.mean(y_train != y_train_pred)
    etest = np.mean(y_test != y_test_pred)
    Ein_CNN.append(ein)
    Etest_CNN.append(etest)
    
print("meanEin of C-NN is {}".format(np.mean(Ein_CNN)))
print("meanEtest of C-NN is {}".format(np.mean(Etest_CNN)))

print("meanEin of K-NN is {}".format(np.mean(Ein_KNN)))
print("meanEtest of K-NN is {}".format(np.mean(Etest_KNN)))

