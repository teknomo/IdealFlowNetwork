# Ideal Flow Python Library
# (c) 2018 Kardi Teknomo
# http://people.revoledu.com/kardi/

import numpy as np
from fractions import gcd, Fraction


def lcm(a,b):
	'''
	return least common multiple of large numbers
	'''
	return a*b // gcd(a,b)


def capacity2adj(C):
	'''
	convert capacity matrix to adjacency matrix
	'''
	return(np.asarray(C)>0).astype(int) # get adjacency matrix structure


def capacity2stochastic(C):
    '''
	convert capacity matrix into stochastic matrix
	'''
    s=np.apply_along_axis(np.sum, axis=1, arr=C)
    return C/s[:,np.newaxis]


def adj2stochastic(A):
	'''
	convert adjacency matrix to stochastic matrix 
	of equal outflow distribution
	'''
	v=np.sum(A,axis=1)           # node out degree
	D=np.diag(v)                 # degree matrix
	return np.dot(np.linalg.inv(D),A) # ideal flow of equal outflow distribution


def idealFlow2stochastic(F):
    '''
	convert ideal flow matrix into Markov stochastic matrix
	''' 
    s=np.apply_along_axis(np.sum, axis=1, arr=F)
    return F/s[:,np.newaxis]

	
def steadyStateMC(S,kappa=1):
    '''
	convert stochastic matrix into steady state Markov vector
    kappa is the total of Markov vector
	'''
    [m,n]=S.shape
    if m==n:
        I=np.eye(n)
        j=np.ones((1,n))
        X=np.concatenate((np.subtract(S.T,I), j), axis=0) # vstack
        Xp=np.linalg.pinv(X)      # Moore-Penrose inverse
        y=np.zeros((m+1,1),float)
        y[m]=kappa
        return np.dot(Xp,y) 


def idealFlow(S,pi):
    '''
	return ideal flow matrix
    based on stochastic matrix and Markov vector
	'''
    return S*pi

	
def adj2idealFlow(A,kappa=1):
	'''
	convert adjacency matrix into ideal flow matrix 
	of equal distribution of outflow 
    kappa is the total flow
	'''
	S=adj2stochastic(A)
	pi=steadyStateMC(S,kappa)
	return idealFlow(S,pi)
	
	
def capacity2idealFlow(C,kappa=1):
    '''
	convert capacity matrix into ideal flow matrix
    kappa is the total flow
	'''
    S=capacity2stochastic(C)
    pi=steadyStateMC(S,kappa)
    return idealFlow(S,pi)

	
def sumOfRow(M):
	'''
	return vector sum of rows
	'''
	[m,n]=M.shape
	j=np.ones((m,1))
	return np.dot(M,j)	


def sumOfCol(M):
	'''
	return row vector sum of columns
	'''
	[m,n]=M.shape
	j=np.ones((1,n))
	return np.dot(j,M)


def isSquare(M):
	'''
	return True if M is a square matrix
	'''
	[m,n]=M.shape
	if m==n:
		return True
	else:
		return False


def isNonNegative(M):
	'''
	return True of M is a non-negative matrix
	'''
	if np.any(M<0):
		return False
	else:
		return True


def isPositive(M):
	'''
	return True of M is a positive matrix
	'''
	if np.any(M<=0):
		return False
	else:
		return True
		

def isPremagic(M):
	'''
	return True if M is premagic matrix
	'''
	sC=sumOfCol(M)
	sR=sumOfRow(M)
	d=np.linalg.norm(np.subtract(sC,sR.T))
	if d<=10000*np.finfo(float).eps:
		return True
	else:
		return False
	

def isIrreducible(M):
	'''
	return True if M is irreducible matrix 
	'''
	if isSquare(M) and isNonNegative(M):
		[m,n]=M.shape
		I=np.eye(n)
		Q=np.linalg.matrix_power(np.add(I,M),n-1) # Q=(I+M)^(n-1)
		return isPositive(Q)
	else:
		return False


def isIdealFlow(M):
	'''
	return True if M is an ideal flow matrix
	'''
	if isNonNegative(M) and isIrreducible(M) and isPremagic(M):
		return True
	else:
		return False


def networkEntropy(S):
	'''
	return the value of network entropy
	'''
	s=S[np.nonzero(S)]
	return np.sum(np.multiply(-s,np.log(s)),axis=None)


def entropyRatio(S):
	'''
	return network entropy ratio
	'''
	h1=networkEntropy(S)
	A=(S>0).astype(int) # get adjacency matrix structure
	T=adj2stochastic(A)
	h0=networkEntropy(T)
	return h1/h0


def globalScaling(F,scalingType='min',val=1):
	'''
	return scaled ideal flow matrix
	input:
	F = ideal flow matrix
	scalingType = {'min','max','sum','int'}
	val = value of the min, max, or sum
	'''
	
	f=F[np.nonzero(F)] # list of non-zero values in F
	if scalingType=='min':
		opt=min(f)
		F=np.multiply(F,val/opt)
	elif scalingType=='max':
		opt=max(f)
		F=np.multiply(F,val/opt)
	elif scalingType=='sum':
		S=idealFlow2stochastic(F)
		pi=steadyStateMC(S,kappa=val)
		F=idealFlow(S,pi)
	elif scalingType=='int':
		denomSet=set()
		for g in f:
			h=Fraction(g).limit_denominator(100000)
			denomSet.add(h.denominator)
		opt=1
		for d in denomSet:
			opt=lcm(opt,d)
		F=np.multiply(F,opt).astype(int)
	else:
		raise ValueError("unknown scalingType")
	return F


if __name__=='__main__':
	C=[[0, 1, 1, 1, 0],   # a
	[0, 0, 0, 1, 0],   # b
   	[0, 1, 1, 0, 0],   # c
   	[0, 0, 0, 0, 2],   # d
   	[1, 0, 1, 2, 0]]
	A=capacity2adj(C)
	S=capacity2stochastic(C)
	F=capacity2idealFlow(C)
	F=globalScaling(F,'int')
	import pandas as pd
	print(pd.DataFrame(F))
