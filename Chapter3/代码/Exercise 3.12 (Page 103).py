# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 09:53:48 2019

@author: qinzhen
"""

import matplotlib.pyplot as plt

#(c)
plt.scatter([-1, 1],[0, 0])
plt.scatter([0, 0],[-1, 1])
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.plot([-2, 2],[0, 0])
plt.plot([0, 0],[-2, 2])
plt.xticks()
plt.show()
