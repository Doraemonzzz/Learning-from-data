# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 11:06:45 2019

@author: qinzhen
"""

import numpy as np
import matplotlib.pyplot as plt

def L(k, x):
    if k == 0:
        return [1]
    elif k == 1:
        return [1, x]
    else:
        temp = L(k-1, x)
        lkx = (2 * k - 1) / k * x * temp[-1] - (k - 1) / k * temp[-2]
        temp.append(lkx)
        return temp

X = np.arange(-1, 1, 0.05)
Y = []
for x in X:
    y = L(6, x)
    Y.append(y)
Y = np.array(Y)

for k in range(7):
    plt.plot(X, Y[:, k], label="L"+str(k)+"(x)")
    plt.legend()

plt.legend()
plt.xlabel("degree")
plt.title("Legendre Polynomial")
plt.show()