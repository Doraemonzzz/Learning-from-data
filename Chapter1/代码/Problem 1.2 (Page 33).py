# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 10:43:54 2019

@author: qinzhen
"""

import matplotlib.pyplot as plt
import numpy as np

x = np.arange(-1, 1, 0.1)
y1 = np.array([-1/3 - 2/3 * i for i in x])
y2 = np.array([-1/3 - 2/3 * i for i in x])
plt.plot(x, y1, label='w=[1,2,3]T')
plt.plot(x, y2, label='w=-[1,2,3]T')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()