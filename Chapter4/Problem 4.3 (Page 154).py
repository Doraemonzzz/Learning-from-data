# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 09:07:19 2018

@author: Administrator
"""

import numpy as np
import matplotlib.pyplot as plt

def L(k,x):
    if(k==0):
        return 1
    elif(k==1):
        return x
    else:
        return (2*k-1)/k*(x*L(k-1,x))-(k-1)/k*L(k-2,x)

x=np.arange(-1,1,0.05)
y=[[] for i in range(6)]
for k in range(6):
    y[k]=[L(k,i) for i in x]
    plt.plot(x,y[k],label="L"+str(k)+"(x)")
plt.legend()
plt.title("Legendre Polynomial")
plt.show()