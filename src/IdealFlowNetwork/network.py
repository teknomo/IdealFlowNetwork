'''

Ideal Flow Network Python Library
version 1.1

@Author: Kardi Teknomo
http://people.revoledu.com/kardi/
(c) 2014-2021 Kardi Teknomo

Last Update: July 14, 2021

Notations:
A = Adjacency matrix
B = Incidence matrix
C = Capacity matrix
F = Flow matrix
S = Stochastic matrix
sR = sum of rows
sC = sum of columns
kappa = total flow
pi = node vector (steady state)
[m,n] = matrix size

'''
import numpy as np
import math
from fractions import Fraction


class IFN():
    def __init__(self, name=""):
        self.name=name      # optional name of the IFN
        
    '''
    
        REUSABLE LIBRARIES
    
    '''
    
    def lcm(self,a,b):
        '''
        return least common multiple of large numbers
        '''
        return a*b // math.gcd(a,b)
    
    
    def hadamardDivision(self, a, b ):
        """ elementwise division by ignore / 0, https://stackoverflow.com/questions/26248654/how-to-return-0-with-divide-by-zero """
        with np.errstate(divide='ignore', invalid='ignore'):
            c = np.true_divide( a, b )
            c[ ~ np.isfinite( c )] = 0  # -inf inf NaN
        return c
    
    '''
    
        MATRIX-TO-MATRIX CONVERSION
    
    '''
    
    def binarizedMatrix(self,M):
        '''
        return (0,1) matrix of M
        '''
        return [[int(bool(x)) for x in l] for l in M]
    
    
    def capacity2adj(self,C):
        '''
        convert capacity matrix to adjacency matrix
        '''
        return(np.asarray(C)>0).astype(int) # get adjacency matrix structure
    
    
    def capacity2stochastic(self,C):
        '''
        convert capacity matrix into stochastic matrix
        S=C./(sR*ones(1,n))
        '''
        s=np.apply_along_axis(np.sum, axis=1, arr=C)
        return C/s[:,np.newaxis]
    
    
    def adj2stochastic(self,A):
        '''
        convert adjacency matrix to stochastic matrix 
        of equal outflow distribution
        '''
        v=np.sum(A,axis=1)           # node out degree
        D=np.diag(v)                 # degree matrix
        return np.dot(np.linalg.inv(D),A) # ideal flow of equal outflow distribution
    
    
    def idealFlow2stochastic(self,F):
        '''
        convert ideal flow matrix into Markov stochastic matrix
        ''' 
        s=np.apply_along_axis(np.sum, axis=1, arr=F)
        return F/s[:,np.newaxis]
    
    
    def idealFlow(self,S,pi):
        '''
        return ideal flow matrix
        based on stochastic matrix and Markov vector
        '''
        # return S.dot(pi)
        [m,n]=S.shape
        jt=np.ones((1,n))
        B=pi.dot(jt)
        return np.multiply(B,S)
    
    
    def adj2idealFlow(self,A,kappa=1):
        '''
        convert adjacency matrix into ideal flow matrix 
        of equal distribution of outflow 
        kappa is the total flow
        '''
        S=self.adj2stochastic(A)
        pi=self.markov(S,kappa)
        return self.idealFlow(S,pi)
        
        
    def capacity2idealFlow(self,C,kappa=1):
        '''
        convert capacity matrix into ideal flow matrix
        kappa is the total flow
        '''
        S=self.capacity2stochastic(C)
        pi=self.markov(S,kappa)
        return self.idealFlow(S,pi)
    
    
    def congestion(self,F,C):
        '''
        return congestion matrix, which is element wise
        division of flow/capacity, except zero remain zero
        
        F,C must be 2D np.array
        '''
        return self.hadamardDivision(F,C)
    
    
      
    '''
    
        MATRIX-TO-VECTOR CONVERSION
    
    '''
    
    def markov(self,S,kappa=1):
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

        
    def sumOfRow(self,M):
        '''
        return vector sum of rows
        '''
        [m,n]=np.array(M).shape
        j=np.ones((m,1))
        return np.dot(M,j)    
    
    
    def sumOfCol(self,M):
        '''
        return row vector sum of columns
        '''
        [m,n]=np.array(M).shape
        j=np.ones((1,n))
        return np.dot(j,M)
    
    
    '''
    
        MATRIX-TESTING
    
    '''
    
    def isSquare(self,M):
        '''
        return True if M is a square matrix
        '''
        [m,n]=np.array(M).shape
        if m==n:
            return True
        else:
            return False
    
    
    def isNonNegative(self,M):
        '''
        return True of M is a non-negative matrix
        '''
        if np.any(np.array(M)<0):
            return False
        else:
            return True
    
    
    def isPositive(self,M):
        '''
        return True of M is a positive matrix
        '''
        if np.any(np.array(M)<=0):
            return False
        else:
            return True
            
    
    def isPremagic(self,M):
        '''
        return True if M is premagic matrix
        '''
        M=np.array(M)
        (n,m)=M.shape
        j=np.ones((n,1))
        sR=np.dot(M,j)
        sC=np.dot(M.transpose(),j)
        return np.allclose(sR,sC)
        
    
    def isIrreducible(self,M):
        '''
        return True if M is irreducible matrix 
        '''
        M=np.array(M)
        if self.isSquare(M) and self.isNonNegative(M):
            [m,n]=M.shape
            I=np.eye(n)
            Q=np.linalg.matrix_power(np.add(I,M),n-1) # Q=(I+M)^(n-1)
            return self.isPositive(Q)
        else:
            return False
    
    
    def isIdealFlow(self,M):
        '''
        return True if M is an ideal flow matrix
        '''
        if self.isNonNegative(M) and self.isIrreducible(M) and self.isPremagic(M):
            return True
        else:
            return False
    
    
    '''
    
        EQUIVALENT IFN
    
    '''
    
    def equivalentIFN(self,F,scaling):
        '''
        return scaled ideal flow matrix
        input:
        F = ideal flow matrix
        scaling = global scaling value
        '''
        F1=F*scaling
        return F1
    
    
    def globalScaling(self,F,scalingType='min',val=1):
        '''
        return scaling factor to ideal flow matrix
        to get equivalentIFN
        input:
        F = ideal flow matrix
        scalingType = {'min','max','sum','int'}
        val = value of the min, max, or sum
        'int' means basis IFN (minimum integer)
        '''
        f=F[np.nonzero(F)] # list of non-zero values in F
        # print('f',f)
        if scalingType=='min':
            opt=min(f)
            scaling=val/opt
        elif scalingType=='max':
            scaling=val/max(f)
        elif scalingType=='sum':
            scaling=val/sum(f)
        elif scalingType=='int':
            denomSet=set()
            for g in f:
                h=Fraction(g).limit_denominator(1000000000)
                denomSet.add(h.denominator)
            scaling=1
            for d in denomSet:
                scaling=self.lcm(scaling,d)
        else:
            raise ValueError("unknown scalingType")
        return scaling
    
    
    '''
    
        GENERATE SPECIAL MATRIX 
    
    '''


    def minIrreducible(self,k):
        '''
        return min irreducible matrix size n by n
        '''
        A=np.zeros((k,k),dtype= np.int8)
        for r in range(k-1):
            c=r+1
            A[r,c]=1
        A[k-1,0]=1
        return A


    def addRandomOnes(self,A,m=6):
        '''
        add 1 to the matrix A at random cell location such that
        the total 1 in the matrix is equal to m
        if total number of 1 is less than m, it will not be added.
        input:
        A = square matrix
        m = number of links (targeted)
        '''
        (n,n1)=A.shape
        n2=np.sum(A,axis=None) # total number of 1 in the matrix
        if m>n2:         # only add 1 if necessary
            k=0         # k is counter of additional 1
            for g in range(n*n):                 # repeat until max (N by N) all filled with 1
                idx=np.random.randint(0, n*n-1)  # create random index
                row=math.ceil(idx/n)         # get row of the random index
                col=((idx-1)%n)+1            # get col of the random index
                if A[row-1,col-1]==0:   # only fill cell that still zero
                    A[row-1,col-1]=1
                    k=k+1
                    if k==m-n2:              # if number of links M has been reached
                        break                # break from for-loop of g (before reaching max)
        return A


    def randIrreducible(self,k=5,m=8):
        '''
        return random irreducible matrix size n by n
        input:
        k = total number of nodes
        m = total number of links >=n
        '''
        A=self.minIrreducible(k)    # create min irreducible matrix
        A1=self.addRandomOnes(A,m)  # add random 1 up to m
        P=self.randPermutationEye(k)  # random permutation of identity matrix
        A2=np.dot(np.dot(P,A1),P.transpose()) # B=P.A.P' shufffle irreducible matrix to remain irreducible
        return A2


    def randPermutationEye(self,n=5):
        '''
        return random permutation matrix of identity matrix size n
        '''
        eye =np.identity(n)
        np.random.shuffle(eye)
        return eye


    '''
    
        INDICES
    
    '''

    def coefVarFlow(self,F):
        '''
        return coeficient variation of the Flow matrix
        '''
        mean=np.mean(F)
        std=np.std(F)
        return mean/std


    def networkEntropy(self,S):
        '''
        return the value of network entropy
        '''
        s=S[np.nonzero(S)]
        return np.sum(np.multiply(-s,np.log(s)),axis=None)
    
    
    def entropyRatio(self,S):
        '''
        return network entropy ratio
        '''
        h1=self.networkEntropy(S)
        A=(S>0).astype(int) # get adjacency matrix structure
        T=self.adj2stochastic(A)
        h0=self.networkEntropy(T)
        return h1/h0    
    
    

if __name__=='__main__':
    import IdealFlowNetwork
    net = IdealFlowNetwork.IFN()
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