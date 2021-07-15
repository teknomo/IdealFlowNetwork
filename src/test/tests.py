# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 12:38:53 2020
IFN Test Suite
@Author: Kardi Teknomo
http://people.revoledu.com/kardi/
"""

import unittest
import numpy as np
from idealflow import network as net

class IFNCoreTestCase(unittest.TestCase):
    def setUp(self):
        self.ifn = net.IFN()
        
        '''
        set up test cases
        '''
        # equal outflow
        self.Adjacency1=[[0, 1, 1, 1, 0],   # a
                         [0, 0, 0, 1, 0],   # b
                         [0, 1, 1, 0, 0],   # c
                         [0, 0, 0, 0, 1],   # d
                         [1, 0, 1, 1, 0]]   # e
        
        self.Stochastic1=[[0, 1/3, 1/3, 1/3, 0],   # a
                         [0, 0, 0, 1, 0],          # b
                         [0, 1/2, 1/2, 0, 0],      # c
                         [0, 0, 0, 0, 1],          # d
                         [1/3, 0, 1/3, 1/3, 0]]    # e
        
        self.Stochastic2=[[0, 1/3, 1/3, 1/3, 0],   # a
                        [0, 0, 0, 1, 0],           # b
                        [0, 1/2, 1/2, 0, 0],       # c
                        [0, 0, 0, 0, 1],           # d
                        [1/4, 0, 1/4, 2/4, 0]      # e
                        ]
        
                      #  a  b  c  d  e 
        self.Capacity2=[[0, 1, 1, 1, 0],   # a
                        [0, 0, 0, 1, 0],   # b
                        [0, 1, 1, 0, 0],   # c
                        [0, 0, 0, 0, 2],   # d
                        [1, 0, 1, 2, 0]    # e
                       ]
        
        self.Flow2=[[0, 1, 1, 1, 0],       # a
                    [0, 0, 0, 5, 0],       # b
                    [0, 4, 4, 0, 0],       # c
                    [0, 0, 0, 0, 12],      # d
                    [3, 0, 3, 6, 0]        # e
                    ]
        
        self.Congestion2=[[0, 1, 1, 1, 0],   # a
                    [0, 0, 0, 5, 0],         # b
                    [0, 4, 4, 0, 0],         # c
                    [0, 0, 0, 0, 6],         # d
                    [3, 0, 3, 3, 0]          # e
                    ]
        
        # random irreducible flow
        k=5
        m=k+int(3*k/4)
        self.FlowRnd=self.ifn.randIrreducible(k,m)
        
        # reducible: absorbing class abc, source class de
        self.AdjReducible1=[[0, 0, 1, 0, 0],# a
                         [1, 0, 0, 0, 0],   # b
                         [0, 1, 0, 0, 0],   # c
                         [0, 1, 0, 0, 1],   # d
                         [0, 0, 0, 1, 0]]   # e
        
        # reducible: absorbing node b
        self.AdjReducible2=[[0, 1, 1, 1, 0],# a
                         [0, 0, 0, 0, 0],   # b
                         [0, 1, 1, 0, 0],   # c
                         [0, 0, 0, 0, 1],   # d
                         [1, 0, 1, 1, 0]]   # e
        
        # reducible: source node a
        self.AdjReducible3=[[0, 1, 1, 1, 0],# a
                         [0, 0, 0, 1, 0],   # b
                         [0, 1, 1, 0, 0],   # c
                         [0, 0, 0, 0, 1],   # d
                         [0, 0, 1, 1, 0]]   # e
        
        # irreducible: high min integer, congestion is row steady
        self.Capacity3=[[0, 5, 0, 8, 1],
                        [6, 0, 0, 10,0],
                        [6, 0, 0, 0, 0],
                        [0, 9, 8, 0, 1],
                        [1, 1, 0, 0, 0]]
        
        self.Capacity4=[[0, 6, 0, 0, 7],
                        [0, 0, 4, 0, 0],
                        [0, 0, 0, 0, 5],
                        [10, 0, 2, 0, 0],
                        [0, 0, 0, 6, 0]]
        
        # this case cannot be solved using Excel Add Ins
        self.Capacity5=[[0, 0, 5, 8, 0],
                        [4, 0, 0, 2, 0],
                        [0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 6],
                        [0, 9, 8, 0, 0]]

            
    # example of random network
    def test_randomNetwork(self):
        net=self.ifn
        net.name="random network"
        
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
        print(pd.DataFrame(F1),'\n')
        
   
    
    '''
    
    STANDARD TESTS 
    
    '''

    
    def testcapacity2adj(self):
        # test capacity2adj
        C=self.Capacity2
        A=self.ifn.capacity2adj(C)
        answer=self.Adjacency1
        retVal=np.array_equal(A,answer)
        self.assertTrue(retVal)
    
    
    def testcapacity2stochastic(self):
        # test capacity2stochastic
        C=self.Capacity2
        S=self.ifn.capacity2stochastic(C)
        answer=self.Stochastic2
        retVal=np.array_equal(S,answer)
        self.assertTrue(retVal)
    
    
    def testCongestion(self):
        C=self.Capacity2
        G=np.array(self.Congestion2)
        F=self.ifn.capacity2idealFlow(C,kappa=40)
        G1=self.ifn.congestion(F,C)
        # print('G=',G1)
        retVal=np.allclose(G,G1)
        self.assertTrue(retVal)
        
    
    def testadj2stochastic(self):
        # test adj2stochastic
        A=self.Adjacency1
        S=self.ifn.adj2stochastic(A)
        answer=self.Stochastic1
        retVal=np.array_equal(S,answer)
        self.assertTrue(retVal)
        
    def testIdealFlow1(self):
        # test steadyStateMC, idealFlow, idealFlow2stochastic
        S=np.array(self.Stochastic1)
        pi=self.ifn.markov(S,kappa=1)
        F=self.ifn.idealFlow(S,pi)
        S1=self.ifn.idealFlow2stochastic(F)
        retVal=np.array_equal(S,S1)
        self.assertTrue(retVal)
    
    def testIdealFlow2(self):
        # test steadyStateMC, idealFlow, idealFlow2stochastic
        S=np.array(self.Stochastic2)
        pi=self.ifn.markov(S,kappa=1)
        F=self.ifn.idealFlow(S,pi)
        S1=self.ifn.idealFlow2stochastic(F)
        retVal=np.array_equal(S,S1)
        self.assertTrue(retVal)
    
    def testIdealFlow3(self):
        # test adj2IdealFlow, capacity2adj
        A=self.Adjacency1
        F=self.ifn.adj2idealFlow(A,kappa=1)
        A1=self.ifn.capacity2adj(F)
        retVal=np.array_equal(A,A1)
        self.assertTrue(retVal)
    
    def testIdealFlow4(self):
        # test capacity2idealFlow
        C=self.Capacity2
        F=self.ifn.capacity2idealFlow(C,kappa=40)
        F1=self.Flow2
        retVal=np.allclose(F,F1)
        self.assertTrue(retVal)
    
    def testIsIrreducible1(self):
        # test randomIrreducible, isIrreducible
        # also test: minIrreducible, addRandomOnes,isPositive
        # randPermutationEye, isSquare, isNonNegative
        n=5
        m=n+int(3*n/4)
        M=self.ifn.randIrreducible(n,m)
        retVal=self.ifn.isIrreducible(M)
        self.assertTrue(retVal)
    
    def testIsIrreducible2(self):
        # test case: reducible class
        A=np.array(self.AdjReducible2)
        retVal=self.ifn.isIrreducible(A)
        self.assertFalse(retVal)
        
    def testIsPremagic1(self):
        # test isPremagic
        F=np.array(self.Flow2)
        retVal=self.ifn.isPremagic(F)
        self.assertTrue(retVal)
        
        
    def testIsIdealFlow1(self):
        # test isIdealFlow positive
        F=np.array(self.Flow2)
        retVal=self.ifn.isIdealFlow(F)
        self.assertTrue(retVal)
    
    def testIsIdealFlow2(self):
        # test isIdealFlow negative
        C=np.array(self.Capacity2)
        retVal=self.ifn.isIdealFlow(C)
        self.assertFalse(retVal)
    
    def testIsIdealFlow3(self):
        # test case: reducible class
        A=self.AdjReducible3
        F=self.ifn.capacity2idealFlow(A,3000)
        # print('F=',F)
        retVal=self.ifn.isIdealFlow(F)
        self.assertFalse(retVal)
        
        
    def testGlobalScaling1(self):
        # test sum global scaling
        kappa=5
        C=np.array(self.Capacity3)
        F=self.ifn.capacity2idealFlow(C,kappa)
        # print('F=',F)
        scaling=self.ifn.globalScaling(F,scalingType='sum',val=kappa)
        # print('scaling:',scaling)
        F1=self.ifn.equivalentIFN(F,scaling)
        # print('F1=',F1)
        retVal=self.ifn.isIdealFlow(F1)
        self.assertTrue(retVal)
        
    def testGlobalScaling2(self):
        # test min global scaling
        kappa=1
        C=np.array(self.Capacity3)
        F=self.ifn.capacity2idealFlow(C,kappa)
        # print('F=',F)
        scaling=self.ifn.globalScaling(F,scalingType='min',val=kappa)
        # print('scaling:',scaling)
        F1=self.ifn.equivalentIFN(F,scaling)
        # print('F1=',F1)
        retVal=self.ifn.isIdealFlow(F1)
        self.assertTrue(retVal)
    
    def testGlobalScaling3(self):
        # test max global scaling
        kappa=1000
        C=np.array(self.Capacity3)
        F=self.ifn.capacity2idealFlow(C,kappa)
        # print('F=',F)
        scaling=self.ifn.globalScaling(F,scalingType='max',val=kappa)
        # print('scaling:',scaling)
        F1=self.ifn.equivalentIFN(F,scaling)
        # print('F1=',F1)
        retVal=self.ifn.isIdealFlow(F1)
        self.assertTrue(retVal)
    
    
    def testGlobalScaling4(self):
        # test int global scaling
        kappa=1
        C=np.array(self.Capacity3)
        F=self.ifn.capacity2idealFlow(C,kappa)
        # print('F=',F)
        scaling=self.ifn.globalScaling(F,scalingType='int',val=kappa)
        # print('scaling:',scaling)
        F1=self.ifn.equivalentIFN(F,scaling)
        # print('F1=',F1)
        retVal=self.ifn.isIdealFlow(F1)
        self.assertTrue(retVal)
    
    def testGlobalScaling5(self):
        # test int global scaling
        kappa=1
        C=np.array(self.Capacity4)
        F=self.ifn.capacity2idealFlow(C,kappa)
        # print('F=',F)
        scaling=self.ifn.globalScaling(F,scalingType='int',val=kappa)
        # print('scaling:',scaling)
        F1=self.ifn.equivalentIFN(F,scaling)
        # print('F1=',F1)
        retVal=self.ifn.isIdealFlow(F1)
        self.assertTrue(retVal)
        
    def testCoefVarFlow(self):
        # test globalScaling min, coefVarFlow
        F=np.array(self.FlowRnd)
        scaling=self.ifn.globalScaling(F,scalingType='min',val=2) # doubling it
        F1=self.ifn.equivalentIFN(F,scaling)
        cv1=self.ifn.coefVarFlow(F)
        cv2=self.ifn.coefVarFlow(F1)
        self.assertEqual(cv1,cv2)
        

if __name__ == '__main__':
    unittest.main() 
