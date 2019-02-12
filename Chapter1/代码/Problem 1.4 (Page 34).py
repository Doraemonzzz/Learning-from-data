# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 11:41:30 2019

@author: qinzhen
"""
import helper as hlp
import numpy as np
import matplotlib.pyplot as plt

#设置随机种子，保证每次结果一致
seed = 42
rnd = np.random.RandomState(seed)

#(a)(b)(c)
N = 20
d = 2
a, b, c, X, y, s, w = hlp.f(N, d, rnd)
hlp.plot_helper(a, b, c, X, y, s, w)

#(d)
N = 100
a, b, c, X, y, s, w = hlp.f(N, d, rnd)
hlp.plot_helper(a, b, c, X, y, s, w)

#(e)
N = 1000
a, b, c, X, y, s, w = hlp.f(N, d, rnd)
hlp.plot_helper(a, b, c, X, y, s, w)

#(f)
#修改数据维度
N = 1000
d = 10
a, b, c, X, y, s, w = hlp.f(N, d, rnd)
print("迭代次数为" + str(s))

#(g)
res = []
for i in range(100):
    a, b, c, X, y, s, w = hlp.f(N, d, rnd, r=0)
    res.append(s)
plt.hist(res, normed=True)
plt.xlabel("迭代次数")
plt.show()
