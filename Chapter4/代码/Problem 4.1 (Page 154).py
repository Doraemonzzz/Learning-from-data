# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 10:51:18 2019

@author: qinzhen
"""

import matplotlib.pyplot as plt
import numpy as np

x = np.arange(-2, 2, 0.01)
degree = np.arange(5)
for i in degree:
    y = x ** i
    label = 'x**'+str(i)
    plt.plot(x, y, label=label)
plt.legend()
plt.show()