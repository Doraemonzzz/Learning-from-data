# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 22:57:08 2019

@author: qinzhen
"""

import matplotlib.pyplot as plt
import numpy as np 

def f1(N, d):
    return (N ** d) + 1

def f2(N, d):
    return (np.exp(1) * N / d) ** d

def draw(d):
    n = range(1,50)
    m1 = [f1(i, d) for i in n]
    m2 = [f2(i, d) for i in n]
    
    plt.plot(n, m1, label='f1')
    plt.plot(n, m2, label='f2')
    plt.legend()
    plt.title('dvc='+str(d))
    plt.show()

#d=2
draw(2)

#d=3
draw(5)