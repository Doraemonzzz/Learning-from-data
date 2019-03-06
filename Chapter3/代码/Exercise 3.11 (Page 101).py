# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 09:51:21 2019

@author: qinzhen
"""

#隐函数作图库
from sympy.parsing.sympy_parser import parse_expr  
from sympy import plot_implicit 
ezplot = lambda exper: plot_implicit(parse_expr(exper))

#a
#wo=0
ezplot('x1**2-x2**2')

#w0>0
ezplot('1+x1**2-x2**2')

#w0<0
ezplot('-1+x1**2-x2**2')

#b
#wo=0
ezplot('x1**2')

#w0>0
ezplot('1+x1**2')

#w0<0
ezplot('-1+x1**2')

#c
ezplot('-1+x1**2+x2**2')

#d
ezplot('1+x1**2+x2**2')