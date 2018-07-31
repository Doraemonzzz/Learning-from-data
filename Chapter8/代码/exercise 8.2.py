# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 01:14:03 2018

@author: Administrator
"""

import matplotlib.pyplot as plt
import numpy as np
w=np.array([1.2,-3.2])
b=-0.5
w1=w/0.5
b1=b/0.5

x=np.array([0,2])
y=-(w[0]*x+b)/w[1]
y1=-(w1[0]*x+b1)/w1[1]

plt.plot(x,y,label='line1')
plt.plot(x,y1,label='line2')
plt.legend()
plt.show()