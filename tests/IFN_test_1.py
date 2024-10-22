# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 12:38:53 2020
last update: Oct 14, 2024

IFN Test Suite - 100% OK
@author: Kardi Teknomo
"""
import unittest
import IdealFlow.Network as net         # import package.module as alias

import matplotlib.pyplot as plt
from unittest.mock import patch
from random import Random
from math import inf
import numpy as np

class IFNCoreTestCase(unittest.TestCase):
    def setUp(self):
        '''
        set up test cases
        
        note: the pyton array below must be converted 
        to np.array before using it with the functions
        '''
        self.ifn = net.IFN("test")
        self.random = Random(1000) # seed
        
        # equal outflow
        self.Adjacency1=[[0, 1, 1, 1, 0],   # a
                         [0, 0, 0, 1, 0],   # b
                         [0, 1, 1, 0, 0],   # c
                         [0, 0, 0, 0, 1],   # d
                         [1, 0, 1, 1, 0]]   # e
        self.Stochastic1=[[0, 1/3, 1/3, 1/3, 0],   # a
                         [0, 0, 0, 1, 0],   # b
                         [0, 1/2, 1/2, 0, 0],   # c
                         [0, 0, 0, 0, 1],   # d
                         [1/3, 0, 1/3, 1/3, 0]]   # e
        
        # generic 
                      #  a  b  c  d  e 
        self.Capacity2=[[0, 1, 1, 1, 0],   # a
                        [0, 0, 0, 1, 0],   # b
                        [0, 1, 1, 0, 0],   # c
                        [0, 0, 0, 0, 2],   # d
                        [1, 0, 1, 2, 0]    # e
                       ]
        self.Stochastic2=[[0, 1/3, 1/3, 1/3, 0],   # a
                        [0, 0, 0, 1, 0],   # b
                        [0, 1/2, 1/2, 0, 0],   # c
                        [0, 0, 0, 0, 1],   # d
                        [1/4, 0, 1/4, 2/4, 0]    # e
                       ]
        self.Flow2=[[0, 1, 1, 1, 0],   # a
                    [0, 0, 0, 5, 0],   # b
                    [0, 4, 4, 0, 0],   # c
                    [0, 0, 0, 0, 12],   # d
                    [3, 0, 3, 6, 0]    # e
                    ]
        self.Congestion2=[[0, 1, 1, 1, 0],   # a
                    [0, 0, 0, 5, 0],   # b
                    [0, 4, 4, 0, 0],   # c
                    [0, 0, 0, 0, 6],   # d
                    [3, 0, 3, 3, 0]    # e
                    ]
        
        # random
        k=5
        m=k+int(3*k/4)
        self.FlowRnd=self.ifn.rand_irreducible(k,m)
        
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
        
    def networkExample1(self):
        # strongly connected, contain cycle
        n=self.ifn
        n.add_link("a","b")
        n.add_link("a","c")
        n.add_link("a","d",77)
        n.add_link("b","c")
        n.add_link("b","e")
        n.add_link("c","e")
        n.add_link("d","c")
        n.add_link("e","a")
        return n
    
    def networkExample2(self):
        # tree, bipartite
        n=self.ifn
        n.add_link("a","b")
        n.add_link("a","c")
        n.add_link("b","d")
        n.add_link("b","e")
        n.add_link("c","f")
        n.add_link("c","g")
        n.add_link("e","h")
        n.add_link("f","i")
        return n
        
    def networkExample3(self):
        # strongly connected (based on tree networkExample2 )
        n=self.ifn
        n.add_link("a","b")
        n.add_link("a","c")
        n.add_link("c","a")
        n.add_link("b","d")
        n.add_link("b","e")
        n.add_link("c","f")
        n.add_link("c","g")
        n.add_link("e","h")
        n.add_link("f","i")    
        n.add_link("d","a")
        n.add_link("h","a")
        n.add_link("i","a")
        n.add_link("g","a")
        return n
    
    def networkExample4(self):
        # premagic
        n=self.ifn
        n.add_link("a","b",2)
        n.add_link("a","c",1)
        n.add_link("a","d",1)
        n.add_link("b","c",1)
        n.add_link("b","e",1)
        n.add_link("c","e",3)
        n.add_link("d","c",1)
        n.add_link("e","a",4)
        return n

    def networkExample5(self):
        # eulerian
        n=self.ifn
        n.add_link("a","b",4)
        n.add_link("b","c",8)
        n.add_link("c","d",8)
        n.add_link("d","a",1)
        n.add_link("a","c",3)
        n.add_link("c","e",5)
        n.add_link("e","a",1)
        return n
    
    def networkExample6(self):
        # flight applications
        n=self.ifn
        n.add_link("New York","Chicago",1000)
        n.add_link("Chicago","Denver",1000)
        n.add_link("New York","Toronto",800)
        n.add_link("New York","Denver",1900)
        n.add_link("Toronto","Calgary",1500)
        n.add_link("Toronto","Los Angeles",1800)
        n.add_link("Toronto","Chicago",500)
        n.add_link("Denver","Urbana",1000)
        n.add_link("Denver","Houston",1500)
        n.add_link("Houston","Los Angeles",1500)
        n.add_link("Denver","Los Angeles",1000)
        return n
    
        
    def networkExample7(self):
        # two components with a bridge
        # contain 2 cycles but not strongly connected
        n=self.ifn
        n.add_link("a","b",4)
        n.add_link("b","c",8)
        n.add_link("c","d",8)
        n.add_link("c","a",1)
        n.add_link("d","e",3)
        n.add_link("e","f",5)
        n.add_link("f","d",1)
        return n
    
    def networkExample8(self):
        # acyclic tree merge in the middle    
        n=self.ifn
        n.add_link("a","b")
        n.add_link("a","c")
        n.add_link("a","d")
        n.add_link("b","e")
        n.add_link("b","f")
        n.add_link("c","f")
        n.add_link("c","g")
        n.add_link("e","h")
        n.add_link("e","i")
        n.add_link("f","j")
        return n
    
    def networkExample9(self):
        # two components of unconnected network
        n=self.ifn
        n.add_link("a","b",4)
        n.add_link("b","c",8)
        n.add_link("c","a",1)
        n.add_link("d","e",3)
        n.add_link("e","f",5)
        n.add_link("f","d",1)
        return n
    
    def networkExample10(self):
        # bipartite
        n=self.ifn
        n.add_link("a","b",1)
        n.add_link("a","d",1)
        n.add_link("b","a",1)
        n.add_link("b","c",1)
        n.add_link("c","b",1)
        n.add_link("c","d",1)
        n.add_link("d","a",1)
        n.add_link("d","c",1)
        return n
    
    def test_network1(self):
        n=self.networkExample1()
        answer={'a': {'b': 1, 'c': 1, 'd': 77}, 'b': {'c': 1, 'e': 1}, 'c': {'e': 1}, 'd': {'c': 1}, 'e': {'a': 1}}
        self.assertEqual(str(n),str(answer))
    
    def test_len1(self):
        n=self.networkExample1()
        answer=5
        self.assertEqual(len(n),answer)
    
    def test_iter1(self):
        n=self.networkExample1()
        lst=[]
        for i in n:  # iterate over start node in network
            lst.append(i)
        answer=['a','b','c','d','e']
        self.assertEqual(lst,answer)
    
    def test_getItem1(self):
        n=self.networkExample1()
        answer=77
        self.assertEqual(n[('a','d')],answer)
        
    def test_setItem1(self):
        # test both setItem then getItem
        n=self.networkExample1()
        answer=75
        n[('a','d')]=answer  # set item
        self.assertEqual(n[('a','d')],answer) # test get item
           
    def test_nodes1(self):
        n=self.networkExample1()
        answer=['a', 'b', 'c', 'd', 'e']
        self.assertEqual(n.nodes,answer)
    
    def test_getMatrix1(self):
        n=self.networkExample1()
        answer=[[0, 1, 1, 77, 0], [0, 0, 1, 0, 1], [0, 0, 0, 0, 1], [0, 0, 1, 0, 0], [1, 0, 0, 0, 0]], ['a', 'b', 'c', 'd', 'e']
        self.assertEqual(n.get_matrix(),answer)
    
    def test_updateLink1(self):
        n=self.networkExample1()
        n.add_link("e","f",5)
        n.add_link("a","b",2)
        answer={'a': {'b': 3, 'c': 1, 'd': 77}, 'b': {'c': 1, 'e': 1}, 'c': {'e': 1}, 'd': {'c': 1}, 'e': {'a': 1, 'f': 5}, 'f':{}}
        self.assertEqual(str(n),str(answer))
    
    def test_add_first_link(self):
        n=self.ifn
        n.cloud_name='#z#'
        n.add_first_link("me","you")
        answer={'#z#': {'me': 1}, 'me': {'you': 1},'you':{}}
        self.assertEqual(str(n),str(answer))
    
    def test_add_last_link(self):
        n=self.ifn
        n.cloud_name='#z#'
        n.add_last_link("me","you")
        answer={'me': {'you': 1}, 'you': {'#z#': 1}, '#z#': {}}
        self.assertEqual(str(n),str(answer))

    def test_set_link_weight1(self):
        n=self.networkExample1()
        n.set_link_weight("a","b",5)
        answer={'a': {'b': 5, 'c': 1, 'd': 77}, 'b': {'c': 1, 'e': 1}, 'c': {'e': 1}, 'd': {'c': 1}, 'e': {'a': 1}}
        self.assertEqual(str(n),str(answer))
    
    def test_set_link_weight_plus_1(self):
        n=self.networkExample1()
        n.set_link_weight_plus_1("a","b")
        answer={'a': {'b': 2, 'c': 1, 'd': 77}, 'b': {'c': 1, 'e': 1}, 'c': {'e': 1}, 'd': {'c': 1}, 'e': {'a': 1}}
        self.assertEqual(str(n),str(answer))
    
    def test_delete_link1(self):
        n=self.networkExample1()
        n.delete_link("a","b")
        answer={'a': {'c': 1, 'd': 77}, 'b': {'c': 1, 'e': 1}, 'c': {'e': 1}, 'd': {'c': 1}, 'e': {'a': 1}}
        self.assertEqual(str(n),str(answer))
    
    def test_delete_link2(self):
        n=self.networkExample1()
        n.delete_link("e","a")
        answer={'a': {'b': 1, 'c': 1, 'd': 77}, 'b': {'c': 1, 'e': 1}, 'c': {'e': 1}, 'd': {'c': 1}}
        self.assertEqual(str(n),str(answer))
    
    def test_getLinks(self):
        n=self.networkExample1()
        answer=[['a','b'],['a','c'],['a','d'],['b','c'],['b','e'],['c','e'],['d','c'],['e','a']]
        self.assertEqual(n.get_links,answer)
    
    def test_totalNodes1(self):
        n=self.networkExample1()
        answer=5
        self.assertEqual(n.total_nodes,answer)
        
    def test_totalLinks1(self):
        n=self.networkExample1()
        answer=8
        self.assertEqual(n.total_links,answer)
    
    def test_density1(self):
        n=self.networkExample1()
        answer=0.8
        self.assertEqual(n.density,answer)
        
    def test_diameter1(self):
        n=self.networkExample1()
        answer=3
        self.assertEqual(n.diameter,answer)
    
    def test_diameter2(self):
        n=self.networkExample2()
        answer=3
        self.assertEqual(n.diameter,answer)
    
    def test_diameter3(self):
        n=self.networkExample3()
        answer=5
        self.assertEqual(n.diameter,answer)
    
    def test_diameter4(self):
        n=self.networkExample4()
        answer=3
        self.assertEqual(n.diameter,answer)
    
    def test_diameter5(self):
        n=self.networkExample5()
        answer=3
        self.assertEqual(n.diameter,answer)    

    def test_diameter6(self):
        n=self.networkExample6()
        answer=3
        self.assertEqual(n.diameter,answer)
    
    def test_diameter7(self):
        n=self.networkExample7()
        answer=5
        self.assertEqual(n.diameter,answer)
    
    def test_diameter8(self):
        n=self.networkExample8()
        answer=3
        self.assertEqual(n.diameter,answer)
        
        
    def test_set_path1(self):
        n=self.networkExample1()
        nodeSequence=['a','a','b','c']
        n.set_path(nodeSequence)
        answer={'a': {'a': 1,'b': 2, 'c': 1, 'd': 77}, 'b': {'c': 2, 'e': 1}, 'c': {'e': 1}, 'd': {'c': 1}, 'e': {'a': 1}}
        self.assertEqual(n.adjList,answer)
    
    def test_path_length1(self):
        n=self.networkExample1()
        path=['a','b','c','e','a']
        answer=2
        self.assertEqual(n.path_length('a','e'),answer)
    
    def test_shortest_path1(self):
        n=self.networkExample1()
        answer=['a','b','e']
        self.assertEqual(n.shortest_path('a','e'),answer)
        
    def test_path_distance1(self):
        n=self.networkExample1()
        answer=2
        self.assertEqual(n.path_distance('a','e'),answer)
        
    
#     # we fixed the seed and make mock functionto make random to deterministic
#     @patch('random.choice', lambda x: x)
#     def test_random_walk_from1(self):
#         n=self.networkExample1()
# #        m = unittest.mock.Mock()
# #        m.n.random_walk_from("a",5)
# #        args, kwargs = m.n.random_walk_from.call_args_list[0]
#         mocked_random_choice = lambda : 0.999#,[0.1,0.1]
#         with patch('random.choice', mocked_random_choice):
#             nodeSequence=n.random_walk_from("a",5)
#         answer=['a', 'd', 'c', 'e', 'a', 'd']
        
#         self.assertEqual(nodeSequence,answer)
    
#     # we fixed the seed and make mock function to make random to deterministic
#     # this still often not working well
#     @patch('random.choice', lambda x: x)
#     def test_random_cycle_from1(self):
#         n=self.networkExample1()
#         mocked_random_choice = lambda : 0.999
#         with patch('random.choice', mocked_random_choice):
#             nodeSequence=n.random_cycle_from("a")
#         answer=['a', 'd', 'c', 'e', 'a']
#         self.assertEqual(nodeSequence,answer)
    
    def test_reindex6(self):
        n=self.networkExample6()
        answer={'Calgary': {}, 'Chicago': {'Denver': 1000}, 'Denver': {'Houston': 1500, 'Los Angeles': 1000, 'Urbana': 1000}, 'Houston': {'Los Angeles': 1500}, 'Los Angeles': {}, 'New York': {'Chicago': 1000, 'Denver': 1900, 'Toronto': 800}, 'Toronto': {'Calgary': 1500, 'Chicago': 500, 'Los Angeles': 1800}, 'Urbana': {}}
        n.reindex()
        self.assertEqual(str(n),str(answer))
        
    def test_totalFlow1(self):
        n=self.networkExample1()
        kappa=n.total_flow
        answer=84
        self.assertEqual(kappa,answer)
    
    def test_network_probability1(self):
        n=self.networkExample1()
        prob=n.network_probability
        answer={'a': {'b': 0.011904761904761904, 'c': 0.011904761904761904, 'd': 0.9166666666666666}, 'b': {'c': 0.011904761904761904, 'e': 0.011904761904761904}, 'c': {'e': 0.011904761904761904}, 'd': {'c': 0.011904761904761904}, 'e': {'a': 0.011904761904761904}}
        # answer={'a': {'b': 0.011904761904761904, 'c': 0.011904761904761904, 'd': 0.9166666666666666}, 'b': {'c': 0.011904761904761904, 'e': 0.011904761904761904}, 'c': {'e': 0.011904761904761904}, 'd': {'c': 0.011904761904761904}, 'e': {'a': 0.011904761904761904}}
        self.assertEqual(prob,answer)
    
    def test_get_reverse_network1(self):
        n=self.networkExample1()
        n1=n.reverse_network()
        answer={'a': {'e': 1}, 'b': {'a': 1}, 'c': {'a': 1, 'b': 1, 'd': 1}, 'd': {'a': 77}, 'e': {'b': 1, 'c': 1}}
        self.assertEqual(str(n1),str(answer))
    
    def test_get_path_probability1(self):
        n=self.networkExample1()
        nodeSequence=['a','b','c','d']
        avgProb,numLink=n.get_path_probability(nodeSequence)
        answer=0.007936507936507936,3  # not strict
        answer=0.0,3  # strict
        self.assertEqual((avgProb,numLink),answer)
    
    def test_get_path_probability2(self):
        # after getting trajectory probability, the kappa mst be the same as original
        n=self.networkExample1()
        nodeSequence=['a','b','c','d']
        avgProb,numLink=n.get_path_probability(nodeSequence)
        kappa=n.total_flow
        answer=84
        self.assertEqual(kappa,answer)
    
    def test_get_path_entropy1(self):
        n=self.networkExample1()
        nodeSequence=['a','b','c','d']
        avgEntropy=n.get_path_entropy(nodeSequence)
        answer=0.05073267795856159
        self.assertEqual(avgEntropy,answer)
    
    def test_dfs1(self):
        n=self.networkExample1()
        nodeSquence=n.dfs("b")
        answer=['b', 'e', 'a', 'd', 'c']
        self.assertEqual(nodeSquence,answer)
    
    def test_dfs3(self):
        n=self.networkExample3()
        nodeSquence=n.dfs("b")
        answer=['b', 'e', 'h', 'a', 'c', 'g', 'f', 'i', 'd']
        self.assertEqual(nodeSquence,answer)
    
    def test_dfs6(self):
        n=self.networkExample6()
        nodeSquence=n.dfs("New York")
        answer=['New York','Denver','Los Angeles','Houston','Urbana', 'Toronto','Chicago','Calgary']
        self.assertEqual(nodeSquence,answer)
    
    def test_dfs7(self):
        n=self.networkExample7()
        nodeSquence=n.dfs("f")
        answer=['f', 'd', 'e']
        self.assertEqual(nodeSquence,answer)
    
    def test_dfs_until1(self):
        n=self.networkExample1()
        nodeSquence=n.dfs_until("b","d")
        answer=['b', 'e', 'a', 'd']
        self.assertEqual(nodeSquence,answer)
    
    def test_dfs_until3(self):
        n=self.networkExample3()
        nodeSquence=n.dfs_until("b",'f')
        answer=['b', 'e', 'h', 'a', 'c', 'g', 'f']
        self.assertEqual(nodeSquence,answer)
        
    def test_dfs_until6(self):
        n=self.networkExample6()
        nodeSquence=n.dfs_until("New York",'Chicago')
        answer=['New York','Denver','Los Angeles','Houston','Urbana', 'Toronto','Chicago']
        self.assertEqual(nodeSquence,answer)
        
    
    def test_is_path1(self):
        n=self.networkExample1()
        nodeSquence=n.backtracking("a","h")
        retVal=n.is_path(nodeSquence)
        self.assertFalse(retVal)
    
    def test_is_path2(self):
        n=self.networkExample1()
        nodeSquence=n.backtracking("a","e")
        retVal=n.is_path(nodeSquence)
        self.assertTrue(retVal)

    def test_is_path3(self):
        n=self.networkExample1()
        nodeSquence=n.dfs("a")
        retVal=n.is_path(nodeSquence)
        self.assertFalse(retVal)
    
    def test_is_trajectory_cycle(self):
        n=self.networkExample7()
        nodeSequence=["a","b","c","a"]
        retVal=n.is_trajectory_cycle(nodeSequence)
        self.assertTrue(retVal)
    
    def test_backtracking1(self):
        n=self.networkExample1()
        nodeSquence=n.backtracking("a","h")
        answer=[]
        self.assertEqual(nodeSquence,answer)
    
    def test_backtracking6(self):
        n=self.networkExample6()
        nodeSquence=n.backtracking("New York","Los Angeles")
        path=['New York', 'Denver', 'Los Angeles']
        self.assertEqual(nodeSquence,path)
        sum=n.path_sum_weight(nodeSquence)
        answer=2900
        self.assertEqual(sum,answer)
    
    def test_backtracking8(self):
        n=self.networkExample8()
        nodeSquence=n.backtracking("a","h")
        answer=['a', 'b', 'e', 'h']
        self.assertEqual(nodeSquence,answer)
    
    def test_find_path1(self):
        n=self.networkExample1()
        nodeSquence=n.find_path("a","h")
        answer=[]
        self.assertEqual(nodeSquence,answer)
    
    def test_find_path7(self):
        n=self.networkExample7()
        nodeSquence=n.find_path("a","f")
        answer=['a', 'b', 'c', 'd', 'e', 'f']
        self.assertEqual(nodeSquence,answer)
        
#    def test_find_path2_1(self):
#        n=self.networkExample1()
#        nodeSquence=n.find_path2("a","h")
#        answer=[]
#        self.assertEqual(nodeSquence,answer)
#    
#    def test_find_path2_7(self):
#        n=self.networkExample7()
#        nodeSquence=n.find_path2("a","f")
#        answer=['a', 'b', 'c', 'd', 'e', 'f']
#        self.assertEqual(nodeSquence,answer)
#        
#    def test_find_path2_8(self):
#        n=self.networkExample8()
#        nodeSquence=n.find_path2("a","h")
#        answer=['a', 'b', 'e', 'h']
#        self.assertEqual(nodeSquence,answer)
        
    def test_find_path8(self):
        n=self.networkExample8()
        nodeSquence=n.find_path("a","h")
        answer=['a', 'b', 'e', 'h']
        self.assertEqual(nodeSquence,answer)
    
    def test_find_all_paths1(self):
        n=self.networkExample1()
        nodeSquence=n.find_all_paths("a","e")
        answer=[['a', 'b', 'c', 'e'], ['a', 'b', 'e'], ['a', 'c', 'e'], ['a', 'd', 'c', 'e']]
        self.assertEqual(nodeSquence,answer)
        
    def test_bfs1(self):
        n=self.networkExample1()
        nodeSquence=n.bfs("b")
        answer=['b', 'c', 'e', 'a', 'd']
        self.assertEqual(nodeSquence,answer)
    
    def test_bfs6(self):
        n=self.networkExample6()
        nodeSquence=n.bfs("New York")
        answer=['New York', 'Chicago', 'Toronto', 'Denver', 'Calgary', 'Los Angeles', 'Urbana', 'Houston']
        self.assertEqual(nodeSquence,answer)
    
    def test_bfs_until1(self):
        n=self.networkExample1()
        nodeSquence=n.bfs_until("b","a")
        answer=['b', 'c', 'e', 'a']
        self.assertEqual(nodeSquence,answer)
    
    def test_bfs_until6(self):
        n=self.networkExample6()
        nodeSquence=n.bfs_until("New York",'Urbana')
        answer=['New York', 'Chicago', 'Toronto', 'Denver', 'Calgary', 'Los Angeles', 'Urbana']
        self.assertEqual(nodeSquence,answer)
    
    def test_all_shortest_path1(self):
        n=self.networkExample1()        
        answer=[[3, 1, 1, 77, 2], 
                [2, 3, 1, 79, 1], 
                [2, 3, 3, 79, 1], 
                [3, 4, 1, 80, 2], 
                [1, 2, 2, 78, 3]]
        matrix,listNode=n.all_shortest_path()
        self.assertEqual(matrix,answer)
       
        
    def test_all_shortest_path6(self):
        n=self.networkExample6()
        
        answer=[[inf, inf, inf, inf, inf, inf, inf, inf],
                [inf, inf, 1000, 2500, 2000, inf, inf, 2000], 
                [inf, inf, inf, 1500, 1000, inf, inf, 1000], 
                [inf, inf, inf, inf, 1500, inf, inf, inf],
                [inf, inf, inf, inf, inf, inf, inf, inf],
                [2300, 1000, 1900, 3400, 2600, inf, 800, 2900], [1500, 500, 1500, 3000, 1800, inf, inf, 2500],
                [inf, inf, inf, inf, inf, inf, inf, inf]]
        matrix,listNode=n.all_shortest_path()
        self.assertEqual(matrix,answer)
    
    
    def test_sumWeightPath1(self):
        n=self.networkExample1()
        path=['a', "b", "d"] # ab=1, bd=0
        answer=1
        self.assertEqual(n.path_sum_weight(path),answer)
    
    def test_sumWeightPath2(self):
        n=self.networkExample1()
        path=['a', "d", "c","a"] # ad=77, dc=1, ca=0
        answer=78
        self.assertEqual(n.path_sum_weight(path),answer)
        
    def test_is_reachable1(self):
        n=self.networkExample1()
        self.assertFalse(n.is_reachable("a","f"))
    
    def test_is_reachable2(self):
        n=self.networkExample2()
        self.assertTrue(n.is_reachable("a","h"))
    
    def test_is_contain_cycle1(self):
        n=self.networkExample1()
        self.assertTrue(n.is_contain_cycle)
    
    def test_is_contain_cycle2(self):
        n=self.networkExample2()
        self.assertFalse(n.is_contain_cycle)
    
    def test_is_contain_cycle7(self):
        n=self.networkExample7()
        self.assertTrue(n.is_contain_cycle)
    
    def test_is_connected1(self):
        n=self.networkExample1()
        self.assertTrue(n.is_connected)
    
    def test_is_connected2(self):
        n=self.networkExample2()
        self.assertTrue(n.is_connected)
    
    def test_is_connected3(self):
        n=self.networkExample3()
        self.assertTrue(n.is_connected)
    
    def test_is_connected4(self):
        n=self.networkExample4()
        self.assertTrue(n.is_connected)
    
    def test_is_connected5(self):
        n=self.networkExample5()
        self.assertTrue(n.is_connected)
    
    def test_is_connected6(self):
        n=self.networkExample6()
        self.assertTrue(n.is_connected)
    
    def test_is_connected7(self):
        n=self.networkExample7()
        self.assertTrue(n.is_connected)
    
    def test_is_connected8(self):
        n=self.networkExample8()
        self.assertTrue(n.is_connected)
        
    def test_is_connected9(self):
        n=self.networkExample9()
        self.assertFalse(n.is_connected)
    
        
    def test_is_strongly_connected1(self):
        n=self.networkExample1()
        self.assertTrue(n.is_strongly_connected)
    
    def test_is_strongly_connected6(self):
        n=self.networkExample6()
        self.assertFalse(n.is_strongly_connected)
    
    def test_is_strongly_connected7(self):
        n=self.networkExample7()
        self.assertFalse(n.is_strongly_connected)
    
    def test_isPremagic1(self):
        n=self.networkExample1()
        self.assertFalse(n.is_premagic)
    
    def test_isPremagic4(self):
        n=self.networkExample4()
        self.assertTrue(n.is_premagic)
    
    def test_isPremagic5(self):
        # Eulerian is not necessarily premagic
        n=self.networkExample5()
        self.assertFalse(n.is_premagic)
    
    def test_isIdealFlow1(self):
        # strongly connected is not necessarily ideal flow
        n=self.networkExample1()
        self.assertFalse(n.is_ideal_flow)
    
    def test_isIdealFlow4(self):
        n=self.networkExample4()
        self.assertTrue(n.is_ideal_flow)
        
    def test_isIdealFlow5(self):
        # Eulerian is not necessarily ideal flow
        n=self.networkExample5()
        self.assertFalse(n.is_ideal_flow)
    
    def test_isEulerianCycle1(self):
        n=self.networkExample1()
        self.assertFalse(n.is_eulerian_cycle)
    
    def test_isEulerianCycle5(self):
        n=self.networkExample5()
        self.assertTrue(n.is_eulerian_cycle)
        
    
    def test_out_weight(self):
        n=self.networkExample1()
        answer=([79, 2, 1, 1, 1],['a', 'b', 'c', 'd', 'e'])
        self.assertEqual(n.out_weight,answer)
    
    def test_in_weight(self):
        n=self.networkExample1()
        answer=([1, 1, 3, 77, 2],['a', 'b', 'c', 'd', 'e'])
        self.assertEqual(n.in_weight,answer)
    
    def test_out_degree(self):
        n=self.networkExample1()
        answer=([3, 2, 1, 1, 1],['a', 'b', 'c', 'd', 'e'])
        self.assertEqual(n.out_degree,answer)
    
    def test_in_degree(self):
        n=self.networkExample1()
        answer=([1, 1, 3, 1, 2],['a', 'b', 'c', 'd', 'e'])
        self.assertEqual(n.in_degree,answer)
    
    def test_is_bipartite1(self):
        n=self.networkExample1()
        self.assertFalse(n.is_bipartite)
    
    def test_is_bipartite2(self):
        n=self.networkExample2()
        self.assertTrue(n.is_bipartite)
    
    def test_is_bipartite3(self):
        n=self.networkExample3()
        self.assertFalse(n.is_bipartite)
        
    def test_is_bipartite10(self):
        n=self.networkExample10()
        self.assertTrue(n.is_bipartite)
    
#    def test_SCC1(self):
#        n=self.networkExample1()
#        n.SCC()
#        
#    def test_SCC3(self):
#        n=self.networkExample3()
#        n.SCC()
    
    def test_out_neighbors1(self):
        n=self.networkExample1()
        answer={'b': 1, 'c': 1, 'd': 77}
        self.assertEqual(n.out_neighbors("a"),answer)
    
    def test_in_neighbors1(self):
        n=self.networkExample1()
        answer={'a': 1, 'b': 1, 'd': 1}
        self.assertEqual(n.in_neighbors("c"),answer)
    
    # def test_updateAdjList1(self):
    #     n=self.networkExample1()
    #     answer={'a': {'b': 5, 'c': 5, 'd': 385}, 'b': {'c': 5, 'e': 5}, 'c': {'e': 5}, 'd': {'c':5}, 'e': {'a': 5}}
    #     self.assertEqual(n.__updateAdjList__(n.adjList,5),answer)
    
    # originally I use this for testing
    def test_randomNetwork1(self):
        net=self.ifn
        net.name="random network"
        plt.close('all')
        k=7
        m=k+int(3*k/4)        
        C=net.rand_irreducible(k,m) # k nodes, m links
        A=net.to_adjacency_matrix(C)
        print("A=",A)
        S=net.capacity_to_stochastic(C)
        print("S=",S)
        F=net.capacity_to_ideal_flow(C)
        print("F=",F)
        scaling=net.global_scaling(F,'int')
        print('scaling:',scaling)
        F1=net.equivalent_ifn(F, scaling)
        net.set_matrix(F1)
        # net.show()
        import pandas as pd
        pd.options.display.float_format = '{:,.0f}'.format
        print(pd.DataFrame(F1))
        
    # def test_show1(self):
    #     n=self.networkExample1()
    #     n.show(layout="Spectral")
    #     n.show(layout="Circular")
    
    # def test_show2(self):
    #     n=self.networkExample2()
    #     n.show(layout="Spectral")
    #     n.show(layout="Kawai")
    #     n.show(layout="Bipartite")
    #     n.show(layout="Random")
    
    # def test_show3(self):
    #     n=self.networkExample3()
    #     n.show(layout="Spring")
    
    # def test_show4(self):
    #     n=self.networkExample4()
    #     n.show(layout="Circular")
    
    # def test_show5(self):
    #     n=self.networkExample5()
    #     n.show(layout="Circular")
    
    # def test_show6(self):
    #     n=self.networkExample6()
    #     n.show(layout="Circular")
        
    # def test_show7(self):
    #     n=self.networkExample7()
    #     n.show(layout="Circular")
        
    # def test_show8(self):
    #     n=self.networkExample8()
    #     n.show(layout="Spectral")
    #     n.show(layout="Bipartite")
        
    # def test_show9(self):
    #     n=self.networkExample9()
    #     n.show(layout="Shell")
        
    # def test_show10(self):
    #     n=self.networkExample10()
    #     n.show(layout="Bipartite")    
    
    '''
    STANDARD TESTS ON Matrix FUNCTIONS
    '''

    def testLCMlist(self):
        # test lcm_ist and lcm
        lst=[2,3,5,15]
        retVal=self.ifn.__lcm_list__(lst)
        answer=30
        self.assertEqual(retVal,answer)
    
    def testcapacity_to_adjacency(self):
        # test capacity_to_adjacency
        C=np.array(self.Capacity2)
        A=self.ifn.to_adjacency_matrix(C)
        answer=np.array(self.Adjacency1)
        retVal=np.array_equal(A,answer)
        self.assertTrue(retVal)
    
    def testcapacity_to_stochastic(self):
        # test capacity_to_stochastic
        C=np.array(self.Capacity2)
        S=self.ifn.capacity_to_stochastic(C)
        answer=self.Stochastic2
        retVal=np.array_equal(S,answer)
        self.assertTrue(retVal)
    
    
    def testCongestion(self):
        C=np.array(self.Capacity2)
        G=np.array(self.Congestion2)
        F=self.ifn.capacity_to_ideal_flow(C,kappa=40)
        G1=self.ifn.congestion(F,C)
        print('G=',G1)
        retVal=np.allclose(G,G1)
        self.assertTrue(retVal)
        
    
    def testadjacency_to_stochastic(self):
        # test adjacency_to_stochastic
        A=np.array(self.Adjacency1)
        S=self.ifn.adjacency_to_stochastic(A)
        answer=np.array(self.Stochastic1)
        retVal=np.array_equal(S,answer)
        self.assertTrue(retVal)
        
    def testIdealFlow1(self):
        # test steadyStateMC, idealFlow, ideal_flow_to_stochastic
        S=np.array(self.Stochastic1)
        pi=self.ifn.markov(S,kappa=1)
        F=self.ifn.ideal_flow(S,pi)
        S1=self.ifn.ideal_flow_to_stochastic(F)
        retVal=np.array_equal(S,S1)
        self.assertTrue(retVal)
    
    def testIdealFlow2(self):
        # test steadyStateMC, idealFlow, ideal_flow_to_stochastic
        S=np.array(self.Stochastic2)
        pi=self.ifn.markov(S,kappa=1)
        F=self.ifn.ideal_flow(S,pi)
        S1=self.ifn.ideal_flow_to_stochastic(F)
        retVal=np.allclose(S,S1)
        self.assertTrue(retVal)
    
    def testIdealFlow3(self):
        # test adjacency_to_ideal_flow, capacity_to_adjacency
        A=np.array(self.Adjacency1)
        F=self.ifn.adjacency_to_ideal_flow(A,kappa=1)
        A1=self.ifn.to_adjacency_matrix(F)
        retVal=np.array_equal(A,A1)
        self.assertTrue(retVal)
    
    def testIdealFlow4(self):
        # test capacity_to_ideal_flow
        C=np.array(self.Capacity2)
        F=self.ifn.capacity_to_ideal_flow(C,kappa=40)
        F1=self.Flow2
        retVal=np.allclose(F,F1)
        self.assertTrue(retVal)
    
    def testis_irreducible1(self):
        # test randomIrreducible, is_irreducible
        # also test: minIrreducible, addRandomOnes,isPositive
        # randPermutationEye, isSquare, isNonNegative
        n=5
        m=n+int(3*n/4)
        M=self.ifn.rand_irreducible(n,m)
        retVal=self.ifn.is_irreducible_matrix(M)
        self.assertTrue(retVal)
    
    def testis_irreducible2(self):
        # test case: reducible class
        A=np.array(self.AdjReducible2)
        retVal=self.ifn.is_irreducible_matrix(A)
        self.assertFalse(retVal)
        
    def testIsPremagic1(self):
        # test isPremagic
        F=np.array(self.Flow2)
        retVal=self.ifn.is_premagic_matrix(F)
        self.assertTrue(retVal)
        
        
    def testIsIdealFlow1(self):
        # test isIdealFlow positive
        F=np.array(self.Flow2)
        retVal=self.ifn.is_ideal_flow_matrix(F)
        self.assertTrue(retVal)
    
    def testIsIdealFlow2(self):
        # test isIdealFlow negative
        C=np.array(self.Capacity2)
        retVal=self.ifn.is_ideal_flow_matrix(C)
        self.assertFalse(retVal)
    
    def testIsIdealFlow3(self):
        # test case: reducible class
        A=self.AdjReducible3
        F=self.ifn.capacity_to_ideal_flow(A,3000)
        # print('F=',F)
        retVal=self.ifn.is_ideal_flow_matrix(F)
        self.assertFalse(retVal)
        
        
    def testglobal_scaling1(self):
        # test sum global scaling
        kappa=5
        C=np.array(self.Capacity3)
        F=self.ifn.capacity_to_ideal_flow(C,kappa)
        # print('F=',F)
        scaling=self.ifn.global_scaling(F,scaling_type='sum',val=kappa)
        # print('scaling:',scaling)
        F1=self.ifn.equivalent_ifn(F,scaling)
        # print('F1=',F1)
        retVal=self.ifn.is_ideal_flow_matrix(F1)
        self.assertTrue(retVal)
        
    def testglobal_scaling2(self):
        # test min global scaling
        kappa=1
        C=np.array(self.Capacity3)
        F=self.ifn.capacity_to_ideal_flow(C,kappa)
        # print('F=',F)
        scaling=self.ifn.global_scaling(F,scaling_type='min',val=kappa)
        # print('scaling:',scaling)
        F1=self.ifn.equivalent_ifn(F,scaling)
        # print('F1=',F1)
        retVal=self.ifn.is_ideal_flow_matrix(F1)
        self.assertTrue(retVal)
    
    def testglobal_scaling3(self):
        # test max global scaling
        kappa=1000
        C=np.array(self.Capacity3)
        F=self.ifn.capacity_to_ideal_flow(C,kappa)
        # print('F=',F)
        scaling=self.ifn.global_scaling(F,scaling_type='max',val=kappa)
        # print('scaling:',scaling)
        F1=self.ifn.equivalent_ifn(F,scaling)
        # print('F1=',F1)
        retVal=self.ifn.is_ideal_flow_matrix(F1)
        self.assertTrue(retVal)
    
    
    def testglobal_scaling4(self):
        # test int global scaling
        print("testglobal_scaling4")
        kappa=1
        C=np.array(self.Capacity3)
        F=self.ifn.capacity_to_ideal_flow(C,kappa)
        print('F=',F)
        scaling=self.ifn.global_scaling(F,scaling_type='int',val=kappa)
        print('scaling:',scaling)
        F1=self.ifn.equivalent_ifn(F,scaling)
        print('F1=',F1)
        retVal=self.ifn.is_ideal_flow_matrix(F1)
        self.assertTrue(retVal)
    
    def testglobal_scaling5(self):
        # test int global scaling
        print("testglobal_scaling5")
        kappa=1
        C=np.array(self.Capacity4)
        F=self.ifn.capacity_to_ideal_flow(C,kappa)
        print('F=',F)
        scaling=self.ifn.global_scaling(F,scaling_type='int',val=kappa)
        print('scaling:',scaling)
        F1=self.ifn.equivalent_ifn(F,scaling)
        print('F1=',F1)
        retVal=self.ifn.is_ideal_flow_matrix(F1)
        self.assertTrue(retVal)
        
    def testcoef_var_flow(self):
        # test global_scaling min, coef_var_flow
        F=np.array(self.FlowRnd)
        scaling=self.ifn.global_scaling(F,scaling_type='min',val=2) # doubling it
        F1=self.ifn.equivalent_ifn(F,scaling)
        cv1=self.ifn.cov_flow_matrix(F)
        cv2=self.ifn.cov_flow_matrix(F1)
        self.assertEqual(cv1,cv2)
    
    def testLowerHigherMarkovOrder1(self):
        # test higher and lower markov order without cloud
        F=np.array(self.FlowRnd)
        self.ifn.set_matrix(F)
        traj=self.ifn.generate('a')
        print(traj)
        trajSuper=self.ifn.order_markov_higher(traj,order=8)
        print('trajSuper\n',trajSuper)
        tr1=self.ifn.order_markov_lower(trajSuper)
        print('traj1\n',tr1)
        self.assertEqual(traj,tr1)
    
    def testLowerHigherMarkovOrder2(self):
        # test higher and lower markov order with cloud
        F=np.array(self.FlowRnd)
        self.ifn.set_matrix(F,['a','b','c','d','#z#'])
        traj=self.ifn.generate('#z#')
        print(traj)
        trajSuper=self.ifn.order_markov_higher(traj,order=8)
        print('trajSuper\n',trajSuper)
        tr1=self.ifn.order_markov_lower(trajSuper)
        print('traj1\n',tr1)
        self.assertEqual(traj,tr1)
    
    def testOvelay1(self):
        net1=net.IFN("test")
        net1.add_link("a","b",1)
        net1.add_link("b","c",1)
        net1.add_link("c","a",1)
        net2=net.IFN("test")
        net2.add_link("a","b",3)
        net2.add_link("b","c",5)
        net2.add_link("c","d",5)
        net2.add_link("d","a",3)
        net2.add_link("d","b",2)
        net3=net.IFN("test")
        net3=net3.overlay(net1, net2) # computed
        net4=net.IFN("test")                # net answer
        net4.add_link("a","b",4)
        net4.add_link("b","c",6)
        net4.add_link("c","a",1)
        net4.add_link("c","d",5)
        net4.add_link("d","a",3)
        net4.add_link("d","b",2)
        self.assertTrue(net4.is_equal_network(net4,net3))

    def testOverlay2(self):
        net1=net.IFN("test")
        net1.add_link("a","d",1)
        net1.add_link("d","f",1)
        net1.add_link("f","a",1)
        net2=net.IFN("test")
        net2.add_link("a","b",3)
        net2.add_link("b","c",5)
        net2.add_link("c","d",5)
        net2.add_link("d","a",3)
        net2.add_link("d","b",2)
        net3=net.IFN("test")
        net3=net3.overlay(net1, net2) # computed
        net4=net.IFN("test")                # net answer
        net4.add_link("a","b",3)
        net4.add_link("a","d",1)
        net4.add_link("b","c",5)
        net4.add_link("c","d",5)
        net4.add_link("d","a",3)
        net4.add_link("d","b",2)
        net4.add_link("d","f",1)
        net4.add_link("f","a",1)
        self.assertTrue(net4.is_equal_network(net4,net3))
    
    def testOverlay3(self):
        net1=net.IFN("test")
        net1.add_link("d","a",1)
        net1.add_link("a","f",1)
        net1.add_link("f","d",1)
        net2=net.IFN("test")
        net2.add_link("a","b",3)
        net2.add_link("b","c",5)
        net2.add_link("c","d",5)
        net2.add_link("d","a",3)
        net2.add_link("d","b",2)
        net3=net.IFN("test")
        net3=net3.overlay(net1, net2) # computed
        net4=net.IFN("test")                # net answer
        net4.add_link("a","b",3)
        net4.add_link("a","f",1)
        net4.add_link("b","c",5)
        net4.add_link("c","d",5)
        net4.add_link("d","a",4)
        net4.add_link("d","b",2)
        net4.add_link("f","d",1)
        self.assertTrue(net4.is_equal_network(net4,net3))
        
    def testUnion1(self):
        net1=net.IFN("test")
        net1.add_link("a","b",1)
        net1.add_link("b","c",1)
        net1.add_link("c","a",1)
        net2=net.IFN("test")
        net2.add_link("a","b",3)
        net2.add_link("b","c",5)
        net2.add_link("c","d",5)
        net2.add_link("d","a",3)
        net2.add_link("d","b",2)
        net3=net.IFN("test")
        net3=net3.union(net1, net2) # computed
        net4=net.IFN("test")                # net answer
        net4.add_link("a","b",3)
        net4.add_link("b","c",5)
        net4.add_link("c","a",1)
        net4.add_link("c","d",5)
        net4.add_link("d","a",3)
        net4.add_link("d","b",2)
        self.assertTrue(net4.is_equal_network(net4,net3))
    
    def testUnion2(self):
        net1=net.IFN("test")
        net1.add_link("a","d",1)
        net1.add_link("d","f",1)
        net1.add_link("f","a",1)
        net2=net.IFN("test")
        net2.add_link("a","b",3)
        net2.add_link("b","c",5)
        net2.add_link("c","d",5)
        net2.add_link("d","a",3)
        net2.add_link("d","b",2)
        net3=net.IFN("test")
        net3=net3.union(net1, net2) # computed
        net4=net.IFN("test")                # net answer
        net4.add_link("a","b",3)
        net4.add_link("a","d",1)
        net4.add_link("b","c",5)
        net4.add_link("c","d",5)
        net4.add_link("d","a",3)
        net4.add_link("d","b",2)
        net4.add_link("d","f",1)
        net4.add_link("f","a",1)
        self.assertTrue(net4.is_equal_network(net4,net3))
        
    def testUnion3(self):
        net1=net.IFN("test")
        net1.add_link("d","a",1)
        net1.add_link("a","f",1)
        net1.add_link("f","d",1)
        net2=net.IFN("test")
        net2.add_link("a","b",3)
        net2.add_link("b","c",5)
        net2.add_link("c","d",5)
        net2.add_link("d","a",3)
        net2.add_link("d","b",2)
        net3=net.IFN("test")
        net3=net3.union(net1, net2) # computed
        net4=net.IFN("test")                # net answer
        net4.add_link("a","b",3)
        net4.add_link("a","f",1)
        net4.add_link("b","c",5)
        net4.add_link("c","d",5)
        net4.add_link("d","a",3)
        net4.add_link("d","b",2)
        net4.add_link("f","d",1)
        self.assertTrue(net4.is_equal_network(net4,net3))
    
    def testEliminate1(self):
        net1=net.IFN("test")
        net1.add_link("a","b",1)
        net1.add_link("b","c",1)
        net1.add_link("c","a",1)
        net4=net.IFN("test")                
        net4.add_link("a","b",4)
        net4.add_link("b","c",6)
        net4.add_link("c","a",1)
        net4.add_link("c","d",5)
        net4.add_link("d","a",3)
        net4.add_link("d","b",2)
        net3=net.IFN("test")
        net3=net3.difference(net4, net1) # computed
        net2=net.IFN("test")           # net answer
        net2.add_link("a","b",3)
        net2.add_link("b","c",5)
        net2.add_link("c","d",5)
        net2.add_link("d","a",3)
        net2.add_link("d","b",2)
        self.assertTrue(net2.is_equal_network(net2,net3))
    
    def testEliminate2(self):
        net2=net.IFN("test")           
        net2.add_link("a","b",3)
        net2.add_link("b","c",5)
        net2.add_link("c","d",5)
        net2.add_link("d","a",3)
        net2.add_link("d","b",2)
        net4=net.IFN("test")                
        net4.add_link("a","b",4)
        net4.add_link("b","c",6)
        net4.add_link("c","a",1)
        net4.add_link("c","d",5)
        net4.add_link("d","a",3)
        net4.add_link("d","b",2)
        net3=net.IFN("test")
        net3=net3.difference(net4, net2) # computed
        net1=net.IFN("test")          # net answer
        net1.add_link("a","b",1)
        net1.add_link("b","c",1)
        net1.add_link("c","a",1)
        self.assertTrue(net1.is_equal_network(net1,net3))
    
    def testEliminate3(self):
        net1=net.IFN("test")
        net1.add_link("a","d",1)
        net1.add_link("d","f",1)
        net1.add_link("f","a",1)
        net4=net.IFN("test")                
        net4.add_link("a","b",3)
        net4.add_link("a","d",1)
        net4.add_link("b","c",5)
        net4.add_link("c","d",5)
        net4.add_link("d","a",3)
        net4.add_link("d","b",2)
        net4.add_link("d","f",1)
        net4.add_link("f","a",1)
        net3=net.IFN("test")
        net3=net3.difference(net4, net1) # computed
        net2=net.IFN("test")              # net answer
        net2.add_link("a","b",3)
        net2.add_link("b","c",5)
        net2.add_link("c","d",5)
        net2.add_link("d","a",3)
        net2.add_link("d","b",2)
        self.assertTrue(net2.is_equal_network(net2,net3))
                        
if __name__=='__main__':
    plt.close('all')
    unittest.main() 
