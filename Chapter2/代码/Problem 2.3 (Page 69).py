# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 22:54:04 2019

@author: qinzhen
"""

#(b)
import matplotlib.pyplot as plt

plt.scatter([1, 2], [0, 0])
plt.scatter([0, 3, 4], [0, 0, 0])
plt.plot([0.5, 0.5], [-1, 1], color='red')
plt.plot([2.5, 2.5], [-1, 1], color='red')
plt.show()

plt.scatter([1, 2], [0, 0])
plt.scatter([3, 4], [0, 0])
plt.plot([0.5, 0.5], [-1, 1], color='red')
plt.plot([2.5, 2.5], [-1, 1], color='red')
plt.show()