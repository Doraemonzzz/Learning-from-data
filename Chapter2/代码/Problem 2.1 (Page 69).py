# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 21:52:33 2019

@author: qinzhen
"""

import numpy as np

delta = 0.03
def f(M, epsilon):
    return np.log(2 * M / delta)/(2 * epsilon ** 2)

#(a)
print(f(1, 0.05))

#(b)
print(f(100, 0.05))

#(c)
print(f(10000, 0.05))