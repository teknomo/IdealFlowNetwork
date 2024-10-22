# Ideal Flow Network Python Library


This Python module is the core library for the computation of *Ideal Flow Network* (IFN). 

### Contents
* [What is IFN?](#What-is-IFN?)
* [How to Install](#How-to-Install)
* [Citing-Ideal-Flow-Network](#Citing-Ideal-Flow-Network)
* [Scientific Basis](#Scientific-Basis)
* [List of Functions](#List-of-Functions)
* [Example](#Example)


## What is IFN?

Ideal Flow is a steady state relative flow distribution in a strongly connected network where the flows are conserved. The IFN theory was first proposed by [Kardi Teknomo](http://people.revoledu.com/kardi/) in 2015. Check also: https://people.revoledu.com/kardi/research/trajectory/ifn/index.html

Ideal Flow is a new concept to analyze transportation networks or communication network. 

For IFN application to traffic assignment check [IFN-Transport](https://github.com/teknomo/ifn-transport) for more details. 


# How to Install

 > **pip install IdealFlowNetwork**

Check Also in [Pypi](https://pypi.org/project/IdealFlowNetwork/)

Latest Stable Version: [1.0.3](https://pypi.org/project/IdealFlowNetwork/1.0.3/)



## Citing Ideal Flow Network
If you use **Ideal Flow Network**, you can cite this paper (Teknomo, K. (2018) Ideal Flow of Markov Chain, Discrete Mathematics, Algorithms and Applications, doi: 10.1142/S1793830918500738 ).

Here is an example BibTeX entry:

```bibtex
@article{doi:10.1142/S1793830918500738,
author = {Teknomo, Kardi},
title = {Ideal flow of Markov Chain},
journal = {Discrete Mathematics, Algorithms and Applications},
volume = {10},
number = {06},
pages = {1850073},
year = {2018},
doi = {10.1142/S1793830918500738},
URL = { 
        https://doi.org/10.1142/S1793830918500738
},
eprint = { 
        https://doi.org/10.1142/S1793830918500738
}}
```

# Scientific Basis
The following publications are the foundations of Ideal Flow analysis:
+ Teknomo, K.(2019), [Ideal Flow Network in Society 5.0](https://link.springer.com/chapter/10.1007/978-3-030-28565-4_11) in Mahdi et al, Optimization in Large Scale Problems - Industry 4.0 and Society 5.0 Applications, Springer, p. 67-69
+ Teknomo, K. and Gardon, R.W. (2019) [Traffic Assignment Based on Parsimonious Data: The Ideal Flow Network](https://ieeexplore.ieee.org/document/8917426), 2019 IEEE Intelligent Transportation Systems Conference (ITSC), 1393-1398.
+ Teknomo, K., Gardon, R. and Saloma, C. (2019), [Ideal Flow Traffic Analysis: A Case Study on a Campus Road Network](https://philjournalsci.dost.gov.ph/images/pdf/pjs_pdf/vol148no1/ideal-flow-trappic-analysis_.pdf), Philippine Journal of Science 148 (1): 5162.
+ Teknomo, K. (2018) [Ideal Flow of Markov Chain](https://www.worldscientific.com/doi/pdf/10.1142/S1793830918500738), Discrete Mathematics, Algorithms and Applications, doi: 10.1142/S1793830918500738 
+ [Teknomo, K.](http://people.revoledu.com/kardi/publication/index.html) and Gardon, R.W. (2017) Intersection Analysis Using the Ideal Flow Model, Proceeding of the IEEE 20th International Conference on Intelligent Transportation Systems, Oct 16-19, 2017, Yokohama, Japan
+ [Teknomo, K.](http://people.revoledu.com/kardi/publication/index.html) (2017) Ideal Relative Flow Distribution on Directed Network, Proceeding of the 12th Eastern Asia Society for Transportation Studies (EASTS), Ho Chi Minh, Vietnam Sept 18-21, 2017.
+ [Teknomo, K.](https://arxiv.org/abs/1706.08856) (2017) Premagic and Ideal Flow Matrices. https://arxiv.org/abs/1706.08856
+ Gardon, R.W. and [Teknomo, K.](http://people.revoledu.com/kardi/publication/index.html) (2017) Analysis of the Distribution of Traffic Density Using the Ideal Flow Method and the Principle of Maximum Entropy, Proceedings of the 17th Philippine Computing Science Congress, Cebu City, March 2017
+ [Teknomo, K.](http://people.revoledu.com/kardi/publication/index.html) (2015) Ideal Flow Based on Random Walk on Directed Graph, The 9th International collaboration Symposium on Information, Production and Systems (ISIPS 2015) 16-18 Nov 2015, Waseda University, KitaKyushu, Japan. 

You may also cite any of the above papers if you use or improve this python library.


<a name="list_functions"></a>
## List of Functions

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

<a name="example"></a>
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
    


More tutorial on Ideal Flow Network is available in [Revoledu.com](http://people.revoledu.com/kardi/tutorial/IFN/) [![@Revoledu](https://img.shields.io/badge/Revol-edu-orange.svg)](http://people.revoledu.com/kardi/tutorial/IFN/). Feel free to join [telegram channel](https://t.me/IdealFlowNetwork/) [![@IdealFlowNetwork](https://img.shields.io/badge/telegram-IdealFlowNetwork-blue.svg)](https://t.me/IdealFlowNetwork/).

## Development

[Pull requests](https://github.com/teknomo/IdealFlowNetwork/pulls) are encouraged and always welcome. Pick an [issue](https://github.com/teknomo/IdealFlowNetwork/issues?q=is%3Aissue+is%3Aopen+sort%3Aupdated-desc) and help us out!




(c) 2021-2023 Kardi Teknomo

