# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 09:39:16 2022

IFN_Core_Tests.py - 100% OK

@author: Kardi Teknomo
"""
import unittest
import IdealFlow.Network as net         # import package.module as alias
import matplotlib.pyplot as plt

class networkOperationsTestCases(unittest.TestCase):
        
    def setUp(self):
        self.isIncludeFailedTest=True
    
    
    '''
            REUSABLE NETWORKS
    '''
    
    def netA1(self,isShow=False):
        '''
        triangle IFN with flow = 1
        '''
        n=net.IFN("test")
        n.add_link("a","b",1)
        n.add_link("b","c",1)
        n.add_link("c","a",1)
        if isShow:
            n.show(layout="Circular")
        return n
    
    
    def netA2(self,isShow=False):
        '''
        triangle IFN with flow = 2
        '''
        n=net.IFN("test")
        n.add_link("a","b",2)
        n.add_link("b","c",2)
        n.add_link("c","a",2)
        if isShow:
            n.show(layout="Circular")
        return n
    
    def netA3(self,isShow=False):
        '''
        triangle IFN with flow = 1 (reverse direction of A5)
        '''
        n=net.IFN("test")
        n.add_link("a","d",1)
        n.add_link("d","f",1)
        n.add_link("f","a",1)
        if isShow:
            n.show(layout="Circular")
        return n
    
    def netA4(self,isShow=False):
        '''
        triangle IFN with flow = 3
        '''
        n=net.IFN("test")
        n.add_link("e","f",3)
        n.add_link("f","g",3)
        n.add_link("g","e",3)
        if isShow:
            n.show(layout="Circular")
        return n
    
    def netA5(self,isShow=False):
        '''
        triangle IFN with flow = 1 (reverse direction of A3)
        '''
        n=net.IFN("test")
        n.add_link("d","a",1)
        n.add_link("a","f",1)
        n.add_link("f","d",1)
        if isShow:
            n.show(layout="Circular")
        return n
    
    def netA6(self,isShow=False):
        '''
        triangle IFN with flow = 1 
        '''
        n=net.IFN("test")
        n.add_link("a","e",1)
        n.add_link("e","f",1)
        n.add_link("f","a",1)
        if isShow:
            n.show(layout="Circular")
        return n
    
    
    def netB1(self,isShow=False):
        '''
        square IFN name B1
        '''
        n=net.IFN("test")
        n.add_link("a","b",3)
        n.add_link("b","c",5)
        n.add_link("c","d",5)
        n.add_link("d","a",3)
        n.add_link("d","b",2)
        if isShow:
            n.show(layout="Circular")
        return n
    
    def netB2(self,isShow=False):
        '''
        square IFN name B2 = A1 + B1
        '''
        n=net.IFN("test")
        n.add_link("a","b",4)
        n.add_link("b","c",6)
        n.add_link("c","a",1)
        n.add_link("c","d",5)
        n.add_link("d","a",3)
        n.add_link("d","b",2)
        if isShow:
            n.show(layout="Circular")
        return n
    
    def netB3(self,isShow=False):
        '''
        square IFN name B3 = A2 + B1
        '''
        n=net.IFN("test")
        n.add_link("a","b",3)
        n.add_link("b","c",5)
        n.add_link("c","d",5)
        n.add_link("d","a",3)
        n.add_link("d","b",2)
        n.add_link("a","b",2)
        n.add_link("b","c",2)
        n.add_link("c","a",2)
        if isShow:
            n.show(layout="Circular")
        return n
    
    
    def netC1(self,isShow=False):
        '''
        house shape IFN name C1 = A3 + B1
        '''
        n=net.IFN("test")
        n.add_link("a","b",3)
        n.add_link("a","d",1)
        n.add_link("b","c",5)
        n.add_link("c","d",5)
        n.add_link("d","a",3)
        n.add_link("d","b",2)
        n.add_link("d","f",1)
        n.add_link("f","a",1)
        if isShow:
            n.show(layout="Circular")
        return n
    
    
    def netC2(self,isShow=False):
        '''
        house shape IFN name C2 = A5 + B1
        '''
        n=net.IFN("test")
        n.add_link("a","b",3)
        n.add_link("b","c",5)
        n.add_link("c","d",5)
        n.add_link("d","a",3)
        n.add_link("d","b",2)
        n.add_link("d","a",1)
        n.add_link("a","f",1)
        n.add_link("f","d",1)
        if isShow:
            n.show(layout="Circular")
        return n
    
    def netD1(self,isShow=False):
        '''
        almost house shape IFN name D1 = A6 + B1
        '''
        n=net.IFN("test")
        n.add_link("a","b",3)
        n.add_link("b","c",5)
        n.add_link("c","d",5)
        n.add_link("d","a",3)
        n.add_link("d","b",2)
        n.add_link("a","e",1)
        n.add_link("e","f",1)
        n.add_link("f","a",1)
        if isShow:
            n.show(layout="Circular")
        return n
    
    
    def netE1(self,isShow=False):
        '''
        unconnected networks name E1 = A4 + B1
        '''
        n=net.IFN("test")
        n.add_link("e","f",3)
        n.add_link("f","g",3)
        n.add_link("g","e",3)
        n.add_link("a","b",3)
        n.add_link("b","c",5)
        n.add_link("c","d",5)
        n.add_link("d","a",3)
        n.add_link("d","b",2)
        
        if isShow:
            n.show(layout="Circular")
        return n
    
    '''
            TESTS
    '''
    
    def test_overlay_2Links(self):
        # test B2 = A1+B1
        A1=self.netA1()
        B1=self.netB1()
        B2=self.netB2() # net answer
        n=net.IFN("test")
        n=n.overlay(A1, B1)      # computed
        self.assertTrue(n.is_equal_network(n,B2),"overlay two networks intersect at of two links")


    def test_overlay_2Links_Reverse(self):
        # test B2 = B1+A1
        A1=self.netA1()
        B1=self.netB1()
        B2=self.netB2() # net answer
        n=net.IFN("test")
        n=n.overlay(B1, A1)      # computed
        self.assertTrue(n.is_equal_network(n,B2),"overlay two networks intersect at of two links, order reverse")

    def test_overlay_2Links_2(self):
        # test B3 = A2+B1
        A2=self.netA2()
        B1=self.netB1()
        B3=self.netB3() # net answer
        n=net.IFN("test")
        n=n.overlay(A2, B1)      # computed
        self.assertTrue(n.is_equal_network(n,B3),"overlay two networks intersect at of two links")

    def test_overlay_2Links_2_Reverse(self):
        # test B3 = B1+A2
        A2=self.netA2()
        B1=self.netB1()
        B3=self.netB3() # net answer
        n=net.IFN("test")
        n=n.overlay(B1, A2)      # computed
        self.assertTrue(n.is_equal_network(n,B3),"overlay two networks intersect at of two links")
    
    def test_difference_2Links(self):
        # test B1 = B2-A1
        A1=self.netA1()
        B1=self.netB1()
        B2=self.netB2() # net answer
        n=net.IFN("test")
        n=n.difference(B2,A1)      # computed
        self.assertTrue(n.is_equal_network(n,B1),"difference two networks intersect at of two links")
    
    def test_difference_2Links_Reverse(self):
        # test A1 = B2-B1
        A1=self.netA1()
        B1=self.netB1()
        B2=self.netB2() # net answer
        n=net.IFN("test")
        n=n.difference(B2, B1)      # computed
        self.assertTrue(n.is_equal_network(n,A1),"difference two networks intersect at of two links")
    
    def test_overlay_2Nodes(self):
        # test C1 = A3+B1
        A3=self.netA3()
        B1=self.netB1()
        C1=self.netC1() # net answer
        n=net.IFN("test")
        n=n.overlay(A3, B1)      # computed
        self.assertTrue(n.is_equal_network(n,C1),"overlay two networks intersect at of two nodes")
    
    def test_overlay_1Nodes(self):
        # test D1 = A6+B1
        A6=self.netA6()
        B1=self.netB1()
        D1=self.netD1() # net answer
        n=net.IFN("test")
        n=n.overlay(A6,B1)      # computed
        self.assertTrue(n.is_equal_network(n,D1),"overlay two networks intersect at of one node")
    
    def test_overlay_Unconnected(self):
        # test E1 = A4+B1
        A4=self.netA4()
        B1=self.netB1()
        E1=self.netE1() # net answer
        n=net.IFN("test")
        n=n.overlay(A4, B1)      # computed
        self.assertTrue(n.is_equal_network(n,E1),"overlay two networks intersect at of two nodes")
    
    def test_complementA1(self):
        A1=self.netA1()
        n=net.IFN("test")
        A1c=n.complement(A1)
        U=n.complete_graph(A1.nodes,weight=A1.max_flow)
        n=n.overlay(A1c, A1)
        self.assertTrue(n.is_equal_network(n,U),'complement overlay: Ac + A == U')
    
    def test_complementOverlayB1(self):
        B1=self.netB1()
        n=net.IFN("test")
        U=n.universe(B1)
        B1c=n.complement(B1)
        n=n.overlay(B1c, B1)
        self.assertTrue(n.is_equal_network(n,U),'complement overlay: Bc + B == U')
    
    '''
            TEST FUNDAMENTAL SET PROPERTIES
    '''
    
    def test_IdempotentUnionB1(self):
        B1=self.netB1()
        n=net.IFN("test")
        n=n.union(B1, B1)
        self.assertTrue(n.is_equal_network(n,B1),'Idempotent Union: B v B = B')
        
    def test_IdempotentIntersectB1(self):
        B1=self.netB1()
        n=net.IFN("test")
        n=n.intersect(B1, B1)
        self.assertTrue(n.is_equal_network(n,B1),'Idempotent Intersect: B ^ B == B')
    
    def test_InvolutionB1(self):
        B1=self.netB1()
        n=net.IFN("test")
        B1c=n.complement(B1)
        n1=n.complement(B1c)
        self.assertTrue(n.is_equal_network(n1,B1),'involution: (Bc)c = B')
    
    def test_DominationUnionB1(self):
        B1=self.netB1()
        n=net.IFN("test")
        U=n.universe(B1)
        n1=n.union(B1,U)
        self.assertTrue(n.is_equal_network(n1,U),'Domination Union: B v U == U')
        
    def test_DominationntersectB1(self):
        B1=self.netB1()
        n=net.IFN("test")
        E=net.IFN("test") # empty
        n1=n.intersect(B1,E)
        self.assertTrue(n.is_equal_network(n1,E),'Domination Intersect: B ^ E == E')
    
    def test_IdentityUnionB1(self):
        B1=self.netB1()
        n=net.IFN("test")
        E=net.IFN("test")   # empty
        n1=n.union(B1,E)
        self.assertTrue(n.is_equal_network(n1,B1),'identiy union: B v E == B')
        
    def test_IdentityIntersectB1(self):
        B1=self.netB1()
        n=net.IFN("test")
        U=n.universe(B1)
        n1=n.intersect(B1,U)
        self.assertTrue(n.is_equal_network(n1,B1),'identity intersect: B ^ U == B')
    
    def test_ComplementU_B1(self):
        B1=self.netB1()
        n=net.IFN("test")
        U=n.universe(B1)
        E=net.IFN("test")
        Uc=n.complement(U)
        self.assertTrue(n.is_equal_network(Uc,E),'complement U based on B: Uc == E (empty)')
        
    def test_ComplementOverlayB1(self):
        B1=self.netB1()
        n=net.IFN("test")
        U=n.universe(B1)
        B1c=n.complement(B1)
        n1=n.overlay(B1, B1c)
        self.assertTrue(n.is_equal_network(n1,U),'complement overlay: B + Bc == U')
    
    @unittest.expectedFailure
    def test_ComplementUnionB1(self):
        B1=self.netB1()
        n=net.IFN("test")
        U=n.universe(B1)
        B1c=n.complement(B1)
        n1=n.union(B1, B1c)
        self.assertTrue(n.is_equal_network(n1,U),'complement union: B v Bc == U')
    
    @unittest.expectedFailure
    def test_deMorganUnionA1B1(self):
        A1=self.netA1()
        B1=self.netB1()
        n=net.IFN("test")
        C1=n.union(A1, B1)
        D1=n.complement(A1)
        E1=n.complement(B1)
        LHS=n.complement(C1)
        RHS=n.intersect(D1,E1)
        self.assertTrue(n.is_equal_network(LHS,RHS),'de Morgan law complement of union: (A v B)c == Ac ^ B^')
    
    @unittest.expectedFailure
    def test_deMorganOverlayA1B1(self):
        A1=self.netA1()
        B1=self.netB1()
        n=net.IFN("test")
        C1=n.overlay(A1, B1)
        D1=n.complement(A1)
        E1=n.complement(B1)
        LHS=n.complement(C1)
        RHS=n.intersect(D1,E1)
        self.assertTrue(n.is_equal_network(LHS,RHS),'de Morgan law complement of overlay: (A + B)c == Ac ^ B^')
    
    @unittest.expectedFailure
    def test_deMorganIntersectA1B1(self):
        A1=self.netA1()
        B1=self.netB1()
        n=net.IFN("test")
        C1=n.intersect(A1, B1)
        LHS=n.complement(C1)
        D1=n.complement(A1)
        E1=n.complement(B1)
        RHS=n.union(D1,E1)
        self.assertTrue(n.is_equal_network(LHS,RHS),'de Morgan law complement of intersection: (A ^ B)c == Ac v Bc')
    
    @unittest.expectedFailure
    def test_deMorganIntersectA1B1_ovelay(self):
        A1=self.netA1()
        B1=self.netB1()
        n=net.IFN("test")
        C1=n.intersect(A1, B1)
        LHS=n.complement(C1)
        D1=n.complement(A1)
        E1=n.complement(B1)
        RHS=n.overlay(D1,E1)
        self.assertTrue(n.is_equal_network(LHS,RHS),'de Morgan law complement of intersection: (A ^ B)c == Ac + Bc')
        
    @unittest.expectedFailure
    def test_ComplementIntersectB1(self):
        B1=self.netB1()
        n=net.IFN("test")
        E=net.IFN("test")
        B1c=n.complement(B1)
        n1=n.intersect(B1, B1c)
        self.assertTrue(n.is_equal_network(n1,E), 'Complement Intersect: B ^ Bc == E')
    
    
    def test_AssociativeUnionA1B1C1(self):
        A1=self.netA1()
        B1=self.netB1()
        C1=self.netC1()
        n1=net.IFN("test")
        n1=n1.union(n1.union(A1, B1),C1)
        n2=net.IFN("test")
        n2=n2.union(A1,n2.union(B1, C1))
        self.assertTrue(n1.is_equal_network(n1,n2),'Associative Union: (A v B) v C == A v (B v C)')
    
    
    def test_AssociativeIntersectA1B1C1(self):
        A1=self.netA1()
        B1=self.netB1()
        C1=self.netC1()
        n1=net.IFN("test")
        n1=n1.intersect(n1.intersect(A1, B1),C1)
        n2=net.IFN("test")
        n2=n2.intersect(A1,n2.intersect(B1, C1))
        self.assertTrue(n1.is_equal_network(n1,n2), 'Associative Intersect: (A ^ B) ^ C == A ^ (B ^ C)')
    
    def test_CommutativeUnionA1B1(self):
        A1=self.netA1()
        B1=self.netB1()
        n1=net.IFN("test")
        n1=n1.union(A1, B1)
        n2=net.IFN("test")
        n2=n2.union(B1, A1)
        self.assertTrue(n1.is_equal_network(n1,n2), 'Commutative Union: A v B == B v A')
    
    def test_CommutativeIntersectA1B1(self):
        A1=self.netA1()
        B1=self.netB1()
        n1=net.IFN("test")
        n1=n1.intersect(A1, B1)
        n2=net.IFN("test")
        n2=n2.intersect(B1, A1)
        self.assertTrue(n1.is_equal_network(n1,n2), 'Commutative Intersect: A ^ B == B ^ A')
    
    
    @unittest.expectedFailure
    def test_setdiffRelation1(self):
        A1=self.netA1()
        B1=self.netB1()
        n1=net.IFN("test")
        AandB=n1.intersect(A1, B1)
        AandBc=n1.complement(AandB)
        n1=n1.union(B1,AandBc)
        n2=net.IFN("test")
        n2=n2.difference(B1,A1)
        self.assertTrue(n1.is_equal_network(n1,n2), 'setdiff Relation: A - B == A v (A ^ B)c')
    
    @unittest.expectedFailure
    def test_setdiffRelation2(self):
        A1=self.netA1()
        B1=self.netB1()
        n1=net.IFN("test")
        AandB=n1.intersect(A1, B1)
        AandBc=n1.complement(AandB)
        n1=n1.overlay(B1,AandBc)
        n2=net.IFN("test")
        n2=n2.difference(B1,A1)
        self.assertTrue(n1.is_equal_network(n1,n2), 'setdiff Relation: A - B == A + (A ^ B)c')
        
        
        
        
    # def test_intersect(self):
    #     A1=self.netA1()
    #     B1=self.netB1()
    #     print('B1',B1)
    #     n=net.IFN("test")
    #     n=n.intersect(A1, B1)
    #     print('Intersection(A1,B1): n',n)
    #     n2=net.IFN("test")
    #     n2=n2.overlay(A1,B1)
    #     print('overlay(A1,B1): n2',n2)
    #     n3=net.IFN("test")
    #     n3=n3.difference(n2,n)
    #     print('Diff(overlay(A1,B1),Intersection(A1,B1)): n3',n3)
    #     B2=self.netB2() # B2 = A1 + B1
    #     print('B2',B2)
    #     self.assertTrue(n.is_equal_network(n2,B2))
                        
        
        
if __name__=='__main__':
    plt.close('all')
    unittest.main() 