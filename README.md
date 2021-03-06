# Ideal Flow Network Python Library


This Python module is the core library for the computation of Ideal Flow Network (IFN). Ideal Flow is a steady state relative flow distribution in a strongly connected network where the flows are conserved. The IFN theory was proposed by [Kardi Teknomo](http://people.revoledu.com/kardi/) in 2015 and subsequently developed by his team. Check also: https://people.revoledu.com/kardi/research/trajectory/ifn/index.html

Ideal Flow is a new concept to analyze transportation networks or communication network. For IFN application to traffic assignment check [IFN-Transport](https://github.com/teknomo/ifn-transport) for more details. 


# How to Install

 > **pip install IdealFlowNetwork**

Check Also in [Pypi](https://pypi.org/project/IdealFlowNetwork/)

Latest Stable Version: [1.0.3](https://pypi.org/project/IdealFlowNetwork/1.0.3/)

### Example


```python
from IdealFlowNetwork import network as ifn

net=ifn.IFN()
net.name="random example network"
        
k=7
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
```

    A= [[0 1 0 0 0 1 0]
     [0 0 0 1 0 0 1]
     [0 0 0 1 0 1 0]
     [0 0 0 0 1 0 0]
     [0 0 1 0 1 0 0]
     [0 0 0 0 0 0 1]
     [1 0 0 0 1 0 0]] 
    
    S= [[0.  0.5 0.  0.  0.  0.5 0. ]
     [0.  0.  0.  0.5 0.  0.  0.5]
     [0.  0.  0.  0.5 0.  0.5 0. ]
     [0.  0.  0.  0.  1.  0.  0. ]
     [0.  0.  0.5 0.  0.5 0.  0. ]
     [0.  0.  0.  0.  0.  0.  1. ]
     [0.5 0.  0.  0.  0.5 0.  0. ]] 
    
    F= [[0.         0.03508772 0.         0.         0.         0.03508772
      0.        ]
     [0.         0.         0.         0.01754386 0.         0.
      0.01754386]
     [0.         0.         0.         0.0877193  0.         0.0877193
      0.        ]
     [0.         0.         0.         0.         0.10526316 0.
      0.        ]
     [0.         0.         0.1754386  0.         0.1754386  0.
      0.        ]
     [0.         0.         0.         0.         0.         0.
      0.12280702]
     [0.07017544 0.         0.         0.         0.07017544 0.
      0.        ]] 
    
    scaling: 57 
    
       0  1  2  3  4  5  6
    0  0  2  0  0  0  2  0
    1  0  0  0  1  0  0  1
    2  0  0  0  5  0  5  0
    3  0  0  0  0  6  0  0
    4  0  0 10  0 10  0  0
    5  0  0  0  0  0  0  7
    6  4  0  0  0  4  0  0
    




# Scientific Basis
The following publications are the foundations of Ideal Flow analysis:

+ Teknomo, K., Gardon, R. and Saloma, C. (2019), Ideal Flow Traffic Analysis: A Case Study on a Campus Road Network, Philippine Journal of Science 148 (1): 5162.
+ Teknomo, K. (2018) Ideal Flow of Markov Chain, Discrete Mathematics, Algorithms and Applications, doi: 10.1142/S1793830918500738 
+ [Teknomo, K.](http://people.revoledu.com/kardi/publication/index.html) and Gardon, R.W. (2017) Intersection Analysis Using the Ideal Flow Model, Proceeding of the IEEE 20th International Conference on Intelligent Transportation Systems, Oct 16-19, 2017, Yokohama, Japan
+ [Teknomo, K.](http://people.revoledu.com/kardi/publication/index.html) (2017) Ideal Relative Flow Distribution on Directed Network, Proceeding of the 12th Eastern Asia Society for Transportation Studies (EASTS), Ho Chi Minh, Vietnam Sept 18-21, 2017.
+ [Teknomo, K.](https://arxiv.org/abs/1706.08856) (2017) Premagic and Ideal Flow Matrices. https://arxiv.org/abs/1706.08856
+ Gardon, R.W. and [Teknomo, K.](http://people.revoledu.com/kardi/publication/index.html) (2017) Analysis of the Distribution of Traffic Density Using the Ideal Flow Method and the Principle of Maximum Entropy, Proceedings of the 17th Philippine Computing Science Congress, Cebu City, March 2017
+ [Teknomo, K.](http://people.revoledu.com/kardi/publication/index.html) (2015) Ideal Flow Based on Random Walk on Directed Graph, The 9th International collaboration Symposium on Information, Production and Systems (ISIPS 2015) 16-18 Nov 2015, Waseda University, KitaKyushu, Japan. 

Please cite any of those papers if you use or improve this python library.


Functions  | Description
---------- | -----------
A = capacity2adj(C) | convert capacity matrix to adjacency matrix
S = capacity2stochastic(C) | convert capacity matrix into stochastic matrix
S = adj2stochastic(A) | convert adjacency matrix to stochastic matrix of equal outflow distribution
S = idealFlow2stochastic(F) | convert ideal flow matrix into Markov stochastic matrix 
pi = steadyStateMC(S,kappa) | convert stochastic matrix into steady state Markov vector. kappa is the total of Markov vector.
F = idealFlow(S,pi) | return ideal flow matrix based on stochastic matrix and Markov vector
F = adj2idealFlow(A,kappa) | convert adjacency matrix into ideal flow matrix of equal distribution of outflow. kappa is the total flow    
F = capacity2idealFlow(C,kappa) | convert capacity matrix into ideal flow vector, kappa is the total flow
sR = sumOfRow(M) | return vector sum of rows of matrix M
sC = sumOfCol(M) | return row vector sum of columns of matrix M
d = isSquare(M) | return True if M is a square matrix
d = isNonNegative(M) | return True of M is a non-negative matrix
d = isPositive(M) | return True of M is a positive matrix
d = isPremagic(M) | return True if M is premagic matrix
d = isIrreducible(M) | return True if M is irreducible matrix
d = isIdealFlow(M) | return True if M is an ideal flow matrix
h = networkEntropy(S) | return the value of network entropy
e = entropyRatio(S) | return network entropy ratio

Tutorial on Ideal Flow Network is available in [Revoledu.com](http://people.revoledu.com/kardi/tutorial/Python/Ideal+Flow.html)

(c) 2021 Kardi Teknomo

