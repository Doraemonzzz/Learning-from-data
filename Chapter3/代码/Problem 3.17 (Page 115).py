# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 14:31:30 2019

@author: qinzhen
"""

import numpy as np
from numpy.linalg import inv

#(b)
def E(u,v):
    return np.exp(u) + np.exp(2 * v) + np.exp(u * v) + u * u - 3 * u * v + 4 * v * v - 3 * u - 5 * v

u = 1 / np.sqrt(13)
v = 3 / (2 * np.sqrt(13))
print(E(u,v))

#(c)
m1 = np.array([[3, -2], [-2, 10]])
m2 = np.array([[2], [3]])
d = inv(m1).dot(m2)

l = np.sqrt((d * d).sum())
d1 = 0.5 * d / l

u = d1[0]
v = d1[1]

print(E(u, v))
print(d1)