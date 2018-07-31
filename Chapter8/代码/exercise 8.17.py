# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 01:18:20 2018

@author: Administrator
"""

import matplotlib.pyplot as plt
import numpy as np

def e1(t):
    return max(1-t,0)

def e2(t):
    if t>=0:
        return 0
    else:
        return 1
    
x=np.arange(-2,2,0.01)
y1=[e1(i) for i in x]
y2=[e2(i) for i in x]

plt.plot(x,y1,label='e1')
plt.plot(x,y2,label='e2')
plt.legend()
plt.title('e1 VS e2')
plt.show()