# -*- coding: utf-8 -*-
"""
sample.py
@Author: Kardi Teknomo
http://people.revoledu.com/kardi/
"""
# from idealflow import network as ifn
import idealflow.network as ifn
net=ifn.IFN()

net.name="random example network"        
k=4
m=k+int(3*k/4)        
C=net.randIrreducible(k,m) # k nodes, m links
A=net.capacity2adj(C)
print("A=",A,'\n')
S=net.capacity2stochastic(C)
print("S=",S,'\n')
F=net.capacity2idealFlow(C)
print("F=",F,'\n')
scaling=net.globalScaling(F,'int')
print('scaling:',scaling,'\n')
F1=net.equivalentIFN(F, scaling)
        
import pandas as pd
pd.options.display.float_format = '{:,.0f}'.format
print(pd.DataFrame(F1))