# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 23:20:11 2019

@author: qinzhen
"""

import numpy as np
import matplotlib.pyplot as plt

delta = 0.05
dvc = 10

def f(N):
    return (8 / N * np.log(4 * ((2 * N) ** dvc + 1) / delta)) ** 0.5 - 0.05

n = 1
while(True):
    if(f(n) <= 0):
        break
    else:
        n += 1

print(n)

#作图
x = range(1, n)
y = [f(i) for i in x]

plt.plot(x, y)
plt.show()