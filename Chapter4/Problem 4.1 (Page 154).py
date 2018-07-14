# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 08:55:37 2018

@author: Administrator
"""

import matplotlib.pyplot as plt
import numpy as np

x=np.arange(-2,2,0.01)
degree=np.arange(5)
for i in degree:
    y=[j**i for j in x]
    label='x**'+str(i)
    plt.plot(x,y,label=label)
plt.legend()
plt.show()