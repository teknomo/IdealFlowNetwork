# -*- coding: utf-8 -*-
"""
Ideal Flow Network 
=====================
@author: Kardi Teknomo (http://people.revoledu.com/kardi/)

IFN Class
    Representation of an expandable Ideal Flow Network.

    The Network class provides various methods for analyzing and manipulating nodes, links, adjacency lists, matrices, 
    and performing network analysis and metric, including path finding, cycles analysis, analyzing connectivity, signature, and more.
    This class is designed to handle flow networks and their properties through mathematical operations and context.    

    Example:
        >>> import IdealFlow.Network as net     # import package.module as alias
        >>> n = net.IFN()
        >>> n.add_link("New York","Chicago",1000)
        >>> n.add_link("Chicago","Denver",1000)
        >>> n.add_link("New York","Toronto",800)
        >>> n.add_link("New York","Denver",1900)
        >>> print(n)
        >>> n.show();

 © 2018-2024 Kardi Teknomo

version 1.15

first build: Sep 3, 2018
last update: Oct 21,2024
"""
import numpy as np
import json
import csv
import math
import matplotlib.pyplot as plt
import networkx as nx
import copy
import json
import random
import heapq


class IFN():
    """
    Represents an Ideal Flow Network (IFN).

    This class provides methods for managing and visualizing a directed flow network.
    """

    def __init__(self, name=""):
        """        
        Initializes a new instance of the IdealFlowNetwork class.

        Parameters:
            name (str): Name of the network. Default is an empty string.

        Attributes:
            version (str): The version of the IFN class.
            name (str): The name of the IFN instance.
            adjList (dict): The adjacency list representing the network structure.
            numNodes (int): The number of nodes in the network.
            listNodes (list): The list of nodes in the network.
            network_prob (dict): A dictionary representing the network's weights (probabilities).
            epsilon (float): A precision constant used for calculations.

        Example:
            >>> import IdealFlow.Network as net     # import package.module as alias
            >>> n = net.IFN()
            >>> print('IFN version:', n.version)
        """
        self.version="1.15"
        self.copyright="© 2018-2024 Kardi Teknomo"
        
        # main two variables
        self.name=name         # name of the IFN
        self.adjList={}        # Adjacency list: key = node, value = dict of neighboring nodes and flows

        # additional object attributes
        self.numNodes=0        # number of nodes
        self.listNodes=[]      # list of nodes
        self.network_prob={}   # new network weight = prob  X (Do we need this?)
        self.cloud_name = "#Z#"

        self.epsilon=0.000001  # precision constant
        
        
    '''
    '
    ' Initialization and Representation
    '
    '''
    

    def __repr__(self):
        """
        Returns Network name

        example: 
        >>> import IdealFlow.Network as net 
        >>> n = net.IFN("test network")
        >>> n
        test network
        
        """
        return self.name
    
    
    def __str__(self):
        """
        Returns a string representation of the Network as Adjacency List
        
        example: 
        >>> import IdealFlow.Network as net 
        >>> n = net.IFN()
        >>> n.add_link('a','b',3)
        >>> print(n)
        {'a': {'b': 3}, 'b': {}}

        """
        return str(self.adjList)
 
    
    def __len__(self):
        """
        Returns the number of nodes in the Network.

        example: 
        >>> import IdealFlow.Network as net 
        >>> n = net.IFN()
        >>> n.add_link('a','b',3)
        >>> len(n)
        2

        Alias: 
            :attr:`total_nodes`

        See also: 
            :attr:`total_nodes`

        """
        # return len(self.__adjList2listNode__(self.adjList))
        return self.numNodes
    
    
    def __iter__(self):
        """
        Returns an iterator over the nodes in the Network.

        Example:
            >>> import IdealFlow.Network as net 
            >>> n = net.IFN()
            >>> n.add_link('a','b')
            >>> for node in n:
            ...     print(node)
            a
            b
        """
        return iter(self.adjList.keys())
    
   
    def __getitem__(self, link):
        """
        Returns the link weight with associated with the given link.

        Parameters:
            link (list of two nodes): [startNode,endNode] 
                use the name or identifier of the node.

        Example:
            >>> import IdealFlow.Network as net 
            >>> n = net.IFN()
            >>> n.add_link('a','b',7)
            >>> n[('a','b')]
            7
            
        """
        (startNode, endNode)=link
        return self.__getWeightLink__(startNode,endNode)
            
    
    def __setitem__(self, link, weight):
        """
        Sets the link weight data for the given link

        Parameters:
            link (list of two nodes): [startNode,endNode] 
                use the name or identifier of the node.

        Example:
            >>> import IdealFlow.Network as net 
            >>> n = net.IFN()
            >>> n.add_link('a','b',7)
            >>> n[('a','b')]=3
            >>> print(n)
            {'a': {'b': 3}, 'b': {}}
        
        """
        (startNode, endNode)=link
        self.__setWeightLink__(startNode,endNode,weight)
    
    
    def set_data(self,adjList):
        """
        Replaces the internal data structure of the network with the given adjacency list.

        This method updates the internal adjacency list representation, the list of nodes, 
        and the number of nodes in the network. The adjacency list represents the network 
        structure where each key corresponds to a node, and the associated value is a list 
        of nodes connected to that key node.

        Parameters:
            adjList (dict): A dictionary representing the adjacency list of the network, 
                            where keys are node identifiers and values are lists of neighboring nodes.
        
        Example:
        >>> import IdealFlow.Network as net 
        >>> n = net.IFN()
        >>> adjList = {'a': {'b': 1, 'c': 3}, 'b': {'c':2}, 'c': {'a':5}}
        >>> n.set_data(adjList)
        >>> print(n)
        {'a': {'b': 1, 'c': 3}, 'b': {'c': 2}, 'c': {'a': 5}}

        Side Effects:
            Updates the following internal attributes:
                - self.adjList: Stores the provided adjacency list.
                - self.listNodes: Stores the list of nodes derived from the adjacency list.
                - self.numNodes: Stores the number of nodes in the updated network.
        """
        self.adjList=adjList                               # replace the adjList
        self.listNodes=self.__adjList2listNode__(adjList)  # replace the listNodes
        self.numNodes=len(self.listNodes)                  # replace the numNodes


    
    def get_data(self):
        """
        Property to return the internal data structure of adjacency list

        Example:
        >>> import IdealFlow.Network as net 
        >>> n = net.IFN()
        >>> adjList = {'a': {'b': 1, 'c': 3}, 'b': {'c':2}, 'c': {'a':5}}
        >>> n.set_data(adjList)
        >>> n.get_data()
        {'a': {'b': 1, 'c': 3}, 'b': {'c':2}, 'c': {'a':5}}
        """
        return self.adjList
    


    '''
    '
    ' Node Management 
    '
    '''

    
    def add_node(self,nodeName):
        """
        Adds a new node to the network if it does not already exist.
        Useful for adding an isolated node

        Parameters:
            node (str): The name or identifier of the node to be added.

        Example:
            >>> import IdealFlow.Network as net 
            >>> n = net.IFN()
            >>> n.add_node('a')
            >>> print('a' in n.listNodes)
            True

        """
        if nodeName not in self.listNodes:
            self.numNodes=self.numNodes+1
            self.listNodes.append(nodeName)
            self.listNodes.sort()
            self.adjList[nodeName]={}
    
    
    def delete_node(self,nodeName):
        """
        Deletes a node from the network and all connected links.

        Parameters:
            node (str): The name or identifier of the node to be deleted.
        
        Example:
            >>> import IdealFlow.Network as net 
            >>> n = net.IFN()
            >>> n.add_link('a','b')
            >>> n.add_link('c','b')
            >>> n.delete_node('b')
            >>> print(n)
        
        """
        if nodeName in self.listNodes:
            self.listNodes.remove(nodeName)
            self.listNodes.sort()
            self.numNodes=self.numNodes-1
            if nodeName in self.adjList:
                del self.adjList[nodeName]
            for startNode in self.adjList.keys():
                self.adjList[startNode].pop(nodeName, None)
    
    
    @property
    def nodes(self):
        """
        Property to returns a list of all nodes in the network.

        Returns:
            list: A list of nodes in the network.

        Example:
            >>> import IdealFlow.Network as net 
            >>> n = net.IFN()
            >>> n.add_link('a','b')
            >>> print(n.nodes)
        
        """
        return self.__adjList2listNode__(self.adjList)
#        return self.listNodes
    
    @property
    def total_nodes(self):
        """
        roperty to returns the total number of nodes in the network.

        Returns:
            int: The total number of nodes.

        Example:
            >>> import IdealFlow.Network as net 
            >>> n = net.IFN()
            >>> n.add_link('a','b')
            >>> print(n.total_nodes)            

        """
#        return len(self.__adjList2listNode__(self.adjList))
        return self.numNodes
    
    
    @property
    def nodes_flow(self):
        """
        Property to returns the flow associated with each node in the network.

        Returns:
            dict: A dictionary where keys are node identifiers and values are their respective flows.
        
            Example:
            >>> import IdealFlow.Network as net 
            >>> n = net.IFN()
            >>> n.add_link('a','b',5)
            >>> print(n.nodes_flow)

        This property is useful for node flow analysis because node flow of an IFN must be the same as pi from markov.
        """
        dicNode={}
        for startNode in self.adjList.keys():
            toNodes=self.out_neighbors(startNode)
            
            lst=[]
            for endNode in toNodes.keys():
                weight=toNodes[endNode]   # only use out_weight
                lst.append(weight)
            dicNode[startNode]=sum(lst)
        return dicNode
    
    
    
    # Link Management Methods
    

    def add_link(self, startNode: str, endNode: str, weight: float = 1) -> None:
        """
        Creates a link between two nodes with the specified weight.
        If the link exists, the weight is updated.

        Parameters:
            startNode (str): The starting node of the link.
            endNode (str): The ending node of the link.
            weight (float): The weight of the link. Defaults to 1.

        Example:
            >>> import IdealFlow.Network as net     # import package.module as alias
            >>> n = net.IFN()
            >>> n.add_link('a', 'b', 5)
            >>> print(n.get_link_flow('a', 'b'))
            5.0
        """  
        # add startNode and endNode if not exist
        if startNode not in self.listNodes:
            self.add_node(startNode)
        if endNode not in self.listNodes:
            self.add_node(endNode)
         
        if startNode in self.adjList.keys(): 
            # if startNode exists in adjList
            toNodes=self.out_neighbors(startNode)
            if endNode in toNodes.keys():
                # if endNode exist, update the link weight
                if weight>0:
                    toNodes[endNode]=toNodes[endNode]+weight
                else:
                    # if after added weight become negative
                    if toNodes[endNode]+weight<=0:
                        self.delete_link(startNode,endNode)
                    else:
                        toNodes[endNode]=toNodes[endNode]+weight
            else:
                # if endNode not exist, 
                if weight>0:
                    # add this endNode only if weight is positive
                    toNodes[endNode]=weight
        else: # if startNode is not yet in adjList
            if weight>0:
                # create this endNode with weight 1
                toNodes={endNode: weight}
        # self.adjList[startNode]=toNodes
    

    def add_first_link(self, startNode: str, endNode: str) -> None:
        """
        Shortcut to add the first link in a training trajectory.

        Parameters:
            startNode (str): The starting node.
            endNode (str): The ending node.

        Example:
            >>> import IdealFlow.Network as net    
            >>> n = net.IFN()
            >>> n.add_first_link('a', 'b')
            >>> print(n.get_links)
            [['#z#', 'a'], ['a', 'b']]
        """
        self.add_link(self.cloud_name,startNode) # cloud to startNode
        self.add_link(startNode,endNode)
    
    
    def add_last_link(self, startNode: str, endNode: str) -> None:
        """
        Shortcut to add the last link in a training trajectory.

        Parameters:
            startNode (str): The starting node.
            endNode (str): The ending node.

        Example:
            >>> import IdealFlow.Network as net 
            >>> n = net.IFN()
            >>> n.add_last_link('a', 'b')
            >>> print(n.get_links)
            [['a', 'b'], ['b', '#z#']]
        """
        self.add_link(startNode,endNode)
        self.add_link(endNode,self.cloud_name) # endNode to cloud
    

    def set_link_weight(self, startNode: str, endNode: str, weight: float) -> None:
        """
        Sets the weight of a link directly. If the link does not exist, it is created.

        Parameters:
            startNode (str): The starting node.
            endNode (str): The ending node.
            weight (float): The weight to set.

        Example:
            >>> import IdealFlow.Network as net  
            >>> n = net.IFN()
            >>> n.set_link_weight('a', 'b', 3)
            >>> print(n.get_link_flow('a', 'b'))
            3.0
        """
        self.__setWeightLink__(startNode,endNode,weight)
    
    
    def set_link_weight_plus_1(self, startNode: str, endNode: str) -> None:
        """
        Increments the weight of a link by 1. If the link does not exist, it is created with weight 1.

        Parameters:
            startNode (str): The starting node.
            endNode (str): The ending node.

        Example:
            >>> import IdealFlow.Network as net  
            >>> n = net.IFN()
            >>> n.set_link_weight_plus_1('a', 'b')
            >>> print(n.get_link_flow('a', 'b'))
            1.0
        """
        self.add_link(startNode, endNode, weight=1)
    

    def get_link_flow(self, startNode: str, endNode: str) -> float:
        """
        Returns the flow of a link between two nodes.

        Parameters:
            startNode (str): The starting node.
            endNode (str): The ending node.

        Returns:
            float: The flow of the link, or NaN if the link does not exist.

        Example:
            >>> import IdealFlow.Network as net 
            >>> n = net.IFN()
            >>> n.add_link('a', 'b', 2)
            >>> print(n.get_link_flow('a', 'b'))
            2.0
        """
        if startNode not in self.listNodes or endNode not in self.listNodes:
            return np.nan
        else:
            toNodes=self.out_neighbors(startNode)
            if endNode in toNodes.keys():
                return toNodes[endNode]
            else:
                return np.nan 
            

    def delete_link(self, startNode: str, endNode: str) -> None:
        """
        Deletes a link between two nodes. If the starting node becomes isolated, it is removed.

        Parameters:
            startNode (str): The starting node.
            endNode (str): The ending node.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> n.add_link('a', 'b', 1)
            >>> n.delete_link('a', 'b')
            >>> print(n.get_link_flow('a', 'b'))
            nan
        """
        try:
            toNodes=self.adjList[startNode]
            del toNodes[endNode]
            toNodes=self.adjList[startNode]
            if toNodes=={}:
                # Remove isolated nodes
                del self.adjList[startNode]
        except KeyError as e:
            print(f"Warning: attempt to delete link `{startNode}{endNode}` but the link start/end with {e} does not exist.")
        except Exception as e:
            raise RuntimeError(f"An error occurred while deleting link: {e}")
        
    
    
    def reduce_link_flow(self, startNode: str, endNode: str) -> None:
        """
        Reduces the flow of a link by 1. If the flow reaches zero or below, the link is deleted.

        Parameters:
            startNode (str): The starting node.
            endNode (str): The ending node.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> n.add_link('a', 'b', 1)
            >>> n.reduce_link_flow('a', 'b')
            >>> print(n.get_link_flow('a', 'b'))
            nan
        """
        self.add_link(startNode, endNode, weight=-1)
        

    @property
    def get_links(self) -> list:
        """
        Returns the list of links in the network.

        Returns:
            list: A list of links, where each link is represented as [startNode, endNode].

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> n.add_link('a', 'b', 1)
            >>> print(n.get_links)
            [['a', 'b']]
        """
        lst=[]
        for startNode in self.adjList.keys(): 
            toNodes=self.adjList[startNode]
            for endNode,weight in toNodes.items():
                lst.append([startNode,endNode])
        return lst
    

    @property
    def total_links(self) -> int:
        """
        Returns the total number of links in the network.

        Returns:
            int: The total number of links.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> n.add_link('a', 'b', 1)
            >>> n.add_link('b', 'c', 2)
            >>> print(n.total_links)
            2
        """
        return len(self.get_links) # get update
    
    
    
    '''
    
        NEIGHBORHOOD
        
    '''
    
    
    def out_neighbors(self, startNode: str) -> dict:
        """
        Return the outgoing neighbors and their weights from the given start node.
        (Successor)

        Parameters:
            startNode (str): The node for which outgoing neighbors are required.

        Returns:
            dict: A dictionary where keys are neighboring nodes and values are edge weights. 
                  Returns an empty dict if the node has no neighbors.

        Example:
            >>> import IdealFlow.Network as net 
            >>> n = net.IFN()
            >>> n.add_link('a', 'b', 3)
            >>> n.add_link('a', 'c', 5)
            >>> n.out_neighbors('a')
            {'b': 3, 'c': 5}
        """
        toNodes={}
        if startNode in self.adjList:
            toNodes=self.adjList[startNode]
        return toNodes
    
    
    def in_neighbors(self, toNode: str) -> dict:
        """
        Return the incoming neighbors and their weights for the given node (predecessors).

        Parameters:
            toNode (str): The node for which incoming neighbors are required.

        Returns:
            dict: A dictionary where keys are incoming nodes and values are edge weights.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> n.add_link('a', 'b', 3)
            >>> n.add_link('c', 'b', 2)
            >>> n.in_neighbors('b')
            {'a': 3, 'c': 2}
        """
        n=self.reverse_network()
        return n.out_neighbors(toNode)
    

    @property
    def out_weight(self) -> tuple:
        """
        Return the total outgoing weight of each node and a list of all nodes.

        Returns:
            tuple: A tuple containing two lists:
                - List of total outgoing weights for each node.
                - List of nodes corresponding to the weights.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> n.add_link('a', 'b', 3)
            >>> n.add_link('a', 'c', 5)
            >>> n.out_weight
            ([8, 0, 0], ['a', 'b', 'c'])
        """
        outWeigh=[]
        vertices=self.nodes
        for startNode in vertices:
            toNodes=self.out_neighbors(startNode)
            sumWeight=0
            for endNode,weight in toNodes.items():
                sumWeight=sumWeight+weight
            outWeigh.append(sumWeight)
        return outWeigh,vertices
    

    @property
    def in_weight(self) -> tuple:
        """
        Return the total incoming weight of each node and a list of all nodes.

        Returns:
            tuple: A tuple containing two lists:
                - List of total incoming weights for each node.
                - List of nodes corresponding to the weights.

        Example:
            >>> import IdealFlow.Network as net 
            >>> n = net.IFN()
            >>> n.add_link('a', 'b', 3)
            >>> n.add_link('c', 'b', 2)
            >>> n.in_weight
            ([0, 5, 0], ['a', 'b', 'c'])
        """
        n=self.reverse_network()
        return n.out_weight
    

    @property
    def out_degree(self) -> tuple:
        """
        Return the out-degree (number of outgoing edges) for each node and a list of all nodes.

        Returns:
            tuple: A tuple containing two lists:
                - List of out-degrees for each node.
                - List of nodes corresponding to the out-degrees.

        Example:
            >>> import IdealFlow.Network as net 
            >>> n = net.IFN()
            >>> n.add_link('a', 'b', 3)
            >>> n.add_link('a', 'c', 2)
            >>> n.out_degree
            ([2, 0, 0], ['a', 'b', 'c'])
        """
        outDeg=[]
        vertices=self.nodes
        for startNode in vertices:
            toNodes=self.out_neighbors(startNode)
            outDeg.append(len(toNodes))
        return outDeg,vertices
    
    @property
    def in_degree(self) -> tuple:
        """
        Return the in-degree (number of incoming edges) for each node and a list of all nodes.

        Returns:
            tuple: A tuple containing two lists:
                - List of in-degrees for each node.
                - List of nodes corresponding to the in-degrees.

        Example:
            >>> import IdealFlow.Network as net 
            >>> n = net.IFN()
            >>> n.add_link('a', 'b', 3)
            >>> n.add_link('a', 'c', 2)
            >>> n.in_degree
            ([0, 1, 1], ['a', 'b', 'c'])
        """
        n=self.reverse_network()
        return n.out_degree
    
    
    '''
    
        NETWORK INDICES
        
    '''
    
    @property
    def density(self) -> float:
        """
        Calculate the density of the graph, which is the ratio of the number of edges to the number of possible edges in a complete graph.
        It measures how close a given graph is to a complete graph.
        The maximal density is 1, if a graph is complete.
        
        Returns:
            float: The density of the graph.

        Example:
            >>> import IdealFlow.Network as net 
            >>> n = net.IFN()
            >>> n.add_link('a', 'b', 3)
            >>> n.add_link('a', 'c', 2)            
            >>> n.density
            0.6666666666666666
        """
        V = self.total_nodes
        E = self.total_links
        return 2.0 * E / (V *(V - 1))
    

    @property
    def diameter(self) -> int:
        """
        Compute the diameter of the network, which is the longest shortest path between any pair of nodes. The diameter d of a graph is defined as 
        the maximum eccentricity of any vertex in the graph. 
        The diameter is the length of the shortest path 
        between the most distanced nodes. 
        To determine the diameter of a graph, 
        first find the shortest path between each pair of vertices. 
        The greatest length of any of these paths is the diameter of the graph.

        Returns:
            int: The diameter of the network.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> n.add_link('a', 'b', 3)
            >>> n.add_link('a', 'c', 2) 
            >>> n.diameter
            1
        """
        v = self.nodes 
        pairs = [ (v[i],v[j]) for i in range(len(v)-1) for j in range(i+1, len(v))]
        smallestPaths = []
        for (s,e) in pairs:
            paths = self.find_all_paths(s,e)
            if paths!=[]:
                smallest = sorted(paths, key=len)[0]
                smallestPaths.append(smallest)
        smallestPaths.sort(key=len)
        # longest path is at the end of list, 
        # i.e. diameter corresponds to the length of this path
        diameter = len(smallestPaths[-1]) - 1
        return diameter
    

    @property
    def total_flow(self) -> float:
        """
        Calculate the total flow in the network, which is the sum of all edge weights.

        Returns:
            float: The total flow in the network.

        Example:
            >>> import IdealFlow.Network as net 
            >>> n = net.IFN()
            >>> n.add_link('a', 'b', 3)
            >>> n.add_link('a', 'c', 2)
            >>> n.total_flow
            5
        """
        kappa=0
        for startNode in self.adjList.keys(): 
            toNodes=self.out_neighbors(startNode)
            for endNode,weight in toNodes.items():
                kappa=kappa+weight
        return kappa


    
     

    '''
    
        STOCHASTIC METHODS
        
    '''
        
    @property
    def row_stochastic(self) -> dict:
        """
        Convert the adjacency list to a row-stochastic form, where values represent the link probability out of each node.

        Returns:
            dict: A row-stochastic adjacency list.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> n.add_link('a', 'b', 3)
            >>> n.add_link('a', 'c', 2)
            >>> n.row_stochastic
            {'a': {'b': 0.6, 'c': 0.4}, 'b': {}, 'c': {}}
        """
        F,listNode=self.get_matrix()
        S=self.ideal_flow_to_stochastic(F)
        n=IFN()
        n.set_matrix(S,listNode)
        return n.adjList 
        
    @property
    def network_probability(self) -> dict:
        """
        Return the adjacency list with link probabilities calculated from the total flow (kappa).

        Returns:
            dict: The adjacency list with link probabilities.

        Example:
            >>> import IdealFlow.Network as net  
            >>> n = net.IFN()
            >>> n.add_link('a', 'b', 3)
            >>> n.add_link('a', 'c', 2)
            >>> n.network_probability
            {'a': {'b': 0.6, 'c': 0.4}, 'b': {}, 'c': {}}
        """
        self.__updateNetworkProbability__()
        return self.network_prob#.adjList



    '''
        TRAJECTORY / PATH METHODS
        
    '''

    def get_path(self, startNode: str, endNode: str) -> list:
        """
        return a path from the startNode to the endNode if exists.
        """
        lst = []
        return self.find_path(startNode,endNode,lst)
    

    def find_path(self, startNode: str, endNode: str, path: list = []) -> list:
        """
        Find a path from the startNode to the endNode using Depth First Search (DFS).

        Parameters:
            startNode (str): The starting node.
            endNode (str): The target node.
            path (list): The current path (used for recursion).

        Returns:
            list: A list of nodes representing the path from startNode to endNode.

        Example:
            >>> import IdealFlow.Network as net     # import package.module as alias
            >>> n = net.IFN()
            >>> n.add_link('a', 'b', 1)
            >>> n.add_link('b', 'c', 1)
            >>> n.find_path('a', 'c')
            ['a', 'b', 'c']

        See also: 
            :meth:`get_path()`
        """
        path = path + [startNode]
        if startNode == endNode:
            return path
        if startNode not in self.adjList:
            return None
        for node in self.adjList[startNode]:
            if node not in path:
                extended_path = self.find_path(node, endNode,path)
                if extended_path: 
                    return extended_path
        return []
    
    # def find_path_cycle_limit(self, startNode, endNode, max_internal_cycle, visited_nodes, cycles):
    #     """
    #     Find a path from startNode to endNode using DFS with backtracking, limiting the number of internal cycles.

    #     Parameters:
    #         startNode (str): The starting node.
    #         endNode (str): The target node.
    #         max_internal_cycle (int): The maximum number of internal cycles allowed.
    #         visited_nodes (set): The set of nodes already visited in the path.
    #         cycles (dict): A dictionary containing the current cycle count.

    #     Returns:
    #         list: A list of nodes forming the path from startNode to endNode.
    #     """
    #     path = []
    #     stack = [(startNode, [startNode], set([startNode]))]

    #     while stack:
    #         current_node, current_path, current_visited = stack.pop()

    #         if current_node == endNode:
    #             # Update the global visited nodes
    #             visited_nodes.update(current_visited)
    #             return current_path

    #         for neighbor in self.adjList.get(current_node, {}):
    #             if neighbor not in visited_nodes:
    #                 stack.append((neighbor, current_path + [neighbor], current_visited | set([neighbor])))
    #             else:
    #                 # Node has been visited; check cycle count
    #                 if cycles['count'] < max_internal_cycle:
    #                     cycles['count'] += 1
    #                     stack.append((neighbor, current_path + [neighbor], current_visited))
    #                 else:
    #                     continue  # Skip this neighbor to limit cycles

    #     return None  # No path found within cycle limit

    def find_path_cycle_limit(self, startNode, endNode, max_internal_cycle, visited_nodes, node_visit_counts=None, path=None):
        """
        Find a path from startNode to endNode using DFS with backtracking, limiting the number of internal cycles.

        Parameters:
            startNode (str): The starting node.
            endNode (str): The target node.
            max_internal_cycle (int): The maximum number of internal cycles allowed.
            visited_nodes (set): The set of nodes already visited in the overall path.
            node_visit_counts (dict): Dictionary of node visit counts in the current recursive path.
            path (list): The current path.

        Returns:
            list: A list of nodes forming the path from startNode to endNode.
        """
        if path is None:
            path = [startNode]
        else:
            path.append(startNode)

        if node_visit_counts is None:
            node_visit_counts = {}

        # Increment the visit count for the current node
        node_visit_counts[startNode] = node_visit_counts.get(startNode, 0) + 1

        # Calculate the allowed visit limit for the node
        if startNode in visited_nodes:
            allowed_visits = max_internal_cycle + 1
        else:
            allowed_visits = 1

        # If we have visited the node more than allowed, backtrack
        if node_visit_counts[startNode] > allowed_visits:
            path.pop()
            node_visit_counts[startNode] -= 1
            return None

        if startNode == endNode:
            # Update the global visited nodes
            visited_nodes.update(path)
            return path.copy()

        if startNode not in self.adjList:
            path.pop()
            node_visit_counts[startNode] -= 1
            return None

        for neighbor in self.adjList[startNode]:
            result = self.find_path_cycle_limit(
                neighbor, endNode, max_internal_cycle, visited_nodes, node_visit_counts, path
            )
            if result is not None:
                return result

        # Backtrack
        path.pop()
        node_visit_counts[startNode] -= 1
        return None

    def find_all_paths(self, startNode: str, endNode: str, path: list = []) -> list:
        """
        Find all possible paths from the startNode to the endNode.

        Parameters:
            startNode (str): The starting node.
            endNode (str): The target node.
            path (list): The current path (used for recursion).

        Returns:
            list: A list of all possible paths from startNode to endNode.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> n.add_link('a', 'b', 1)
            >>> n.add_link('b', 'c', 1)
            >>> n.find_all_paths('a', 'c')
            [['a', 'b', 'c']]
        """
        path = path + [startNode]
        if startNode == endNode:
            return [path]
        if startNode not in self.adjList:
            return []
        paths = []
        for node in self.adjList[startNode]:
            if node not in path:
                extended_paths = self.find_all_paths(node,endNode,path)
                for p in extended_paths: 
                    paths.append(p)
        return paths
    
    
    def shortest_path(self, startNode: str, endNode: str) -> list:
        """
        Find the shortest path (minimum number of links) between startNode and endNode.

        Parameters:
            startNode (str): The starting node.
            endNode (str): The target node.

        Returns:
            list: The shortest path from startNode to endNode.

        Example:
            >>> import IdealFlow.Network as net  
            >>> n = net.IFN()
            >>> n.add_link('a', 'b', 1)
            >>> n.add_link('b', 'c', 1)
            >>> n.shortest_path('a', 'c')
            ['a', 'b', 'c']
        """
        paths = self.find_all_paths(startNode,endNode)
        if paths!=[]:
            shortest = sorted(paths, key=len)[0]
        else:
            shortest = []
        return shortest
    
    
    def all_shortest_path(self) -> tuple:
        """
        Return the matrix of all shortest paths using the Floyd-Warshall algorithm.
        note: this is min weight path, not min number of links.
        
        Returns:
            tuple: A tuple containing the shortest path matrix and the list of nodes.

        Example:
            >>> import IdealFlow.Network as net     # import package.module as alias
            >>> n = net.IFN()
            >>> n.add_link('a', 'b', 1)
            >>> n.add_link('b', 'c', 1)
            >>> n.all_shortest_path()
            (array([[0., 1., 2.], [inf, 0., 1.], [inf, inf, 0.]]), ['a', 'b', 'c'])
        """
        # prepare the dist matrix with inf to replace 0
        m,listNode=self.get_matrix()
        n=len(listNode)
        m=IFN.matrix_replace_value(m,0,math.inf) # replace zero with math.inf
        
        for k in range(n):
            d = [list(row) for row in m] # make a copy of distance matrix
            for i in range(n):
                for j in range(n):
                    # Choose if the k vertex can work as a path with shorter distance
                    d[i][j] = min(m[i][j], m[i][k] + m[k][j])
            m=d
        return m,listNode
    

    def min_flow_path(self, startNode, endNode):
        """
        Find the path from startNode to endNode with the minimum total flow (sum of link weights).

        Parameters:
            startNode (str): The starting node.
            endNode (str): The ending node.

        Returns:
            tuple: A tuple containing the minimum flow and the path as a list of nodes.

        Example:
            >>> import IdealFlow.Network as net     # import package.module as alias
            >>> n = net.IFN()
            >>> adjList = {
                    'a': {'c': 1, 'd': 2},
                    'b': {'c': 1, 'e': 3},
                    'c': {'e': 5},
                    'd': {'c': 5},
                    'e': {'a': 3},
                    '#Z#': {'a': 0, 'b': 0}  # Cloud node connections
                }
            >>> n.set_data(adjList)
            >>> min_flow, min_path = n.min_flow_path('a', 'e')
            >>> print(f"Minimum flow from 'a' to 'e': {min_flow}, Path: {min_path}")
            
        """
        
        # Initialize distances and predecessor dictionaries
        distances = {node: float('inf') for node in self.listNodes}
        predecessors = {node: None for node in self.listNodes}
        distances[startNode] = 0

        # Priority queue to select the next node with the smallest distance
        queue = [(0, startNode)]

        while queue:
            current_distance, current_node = heapq.heappop(queue)

            if current_node == endNode:
                break

            for neighbor, flow in self.adjList[current_node].items():
                distance = current_distance + flow
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    predecessors[neighbor] = current_node
                    heapq.heappush(queue, (distance, neighbor))

        # Reconstruct path
        path = []
        node = endNode
        while node is not None:
            path.insert(0, node)
            node = predecessors[node]

        return distances[endNode], path


    def max_flow_path(self, startNode, endNode):
        """
        Find the path from startNode to endNode with the maximum total flow (sum of link weights).

        Parameters:
            startNode (str): The starting node.
            endNode (str): The ending node.

        Returns:
            tuple: A tuple containing the maximum flow and the path as a list of nodes.

        Example:
            >>> import IdealFlow.Network as net     # import package.module as alias
            >>> n = net.IFN()
            >>> adjList = {
                    'a': {'c': 1, 'd': 2},
                    'b': {'c': 1, 'e': 3},
                    'c': {'e': 5},
                    'd': {'c': 5},
                    'e': {'a': 3},
                    '#Z#': {'a': 0, 'b': 0}  # Cloud node connections
                }
            >>> n.set_data(adjList)
            >>> max_flow, max_path = network.max_flow_path('a', 'e')
            >>> print(f"Maximum flow from 'a' to 'e': {max_flow}, Path: {max_path}")

        """        

        # Initialize distances and predecessor dictionaries
        distances = {node: float('-inf') for node in self.listNodes}
        predecessors = {node: None for node in self.listNodes}
        distances[startNode] = 0

        # Priority queue to select the next node with the largest distance
        queue = [(-0, startNode)]

        while queue:
            current_distance, current_node = heapq.heappop(queue)
            current_distance = -current_distance

            if current_node == endNode:
                break

            for neighbor, flow in self.adjList[current_node].items():
                distance = current_distance + flow
                if distance > distances[neighbor]:
                    distances[neighbor] = distance
                    predecessors[neighbor] = current_node
                    heapq.heappush(queue, (-distance, neighbor))

        # Reconstruct path
        path = []
        node = endNode
        while node is not None:
            path.insert(0, node)
            node = predecessors[node]

        return distances[endNode], path

   
    def query(self, sequence, method='min'):
        """
        Find a cycle path from a cloud node to a cloud node that contains the node sequence.

        Parameters:
            sequence (list or str): The sequence of nodes to include in the path.
            method (str): 'min' for minimum flow paths, 'max' for maximum flow paths.

        Returns:
            tuple: A tuple containing the path as a list of nodes and the probability.

        Example:
            >>> path, probability = network.query('ace', method='min')
            >>> print(f"Query result: Probability: {probability}, Path: {path}")
        """
        if isinstance(sequence, str):
            sequence = list(sequence)

        cloud_node = self.cloud_name  # The cloud node identifier

        # Step 1: Find path from cloud node to start node of the sequence
        start_node = None
        for node in sequence:
            if node in self.listNodes:
                start_node = node
                break
        if not start_node:
            return 0, []  # No nodes in sequence exist in the network

        # Path from cloud node to start_node
        if method == 'min':
            _, path1 = self.min_flow_path(cloud_node, start_node)
        elif method == 'max':
            _, path1 = self.max_flow_path(cloud_node, start_node)
        elif method == 'random':
            length = np.random.randint(1,int(self.numNodes/5))
            path1 = self.random_walk_from(cloud_node, length, allow_internal_cycles=False, max_internal_cycles=1)
            # path1.append(start_node)
        else:
            path1 = self.get_path(cloud_node, start_node)
        if not path1:
            return 0, []  # No path from cloud to start node

        # Step 2: Build path through sequence
        sequence_path = []
        i = sequence.index(start_node)
        sequence_path.append(start_node)
        while i < len(sequence) - 1:
            node1 = sequence[i]
            node2 = sequence[i + 1]
            if node2 not in self.listNodes:
                i += 1
                continue  # Skip nodes not in network
            if method == 'min':
                _, path_seq = self.min_flow_path(node1, node2)
            elif method == 'max':
                _, path_seq = self.max_flow_path(node1, node2)
            else:
                path_seq = self.get_path(node1, node2)
            if not path_seq:
                i += 1
                continue  # Skip if no path between nodes
            sequence_path.extend(path_seq[1:])  # Exclude first node to avoid duplication
            i += 1

        if not sequence_path:
            return 0, []  # No valid path through sequence

        # Step 3: Find path from end of sequence to cloud node
        end_node = None
        for node in reversed(sequence):
            if node in self.listNodes:
                end_node = node
                break
        if not end_node:
            return 0, []  # No nodes in sequence exist in the network

        if method == 'min':
            _, path3 = self.min_flow_path(end_node, cloud_node)
        elif method == 'max':
            _, path3 = self.max_flow_path(end_node, cloud_node)
        elif method == 'random':
            length = np.random.randint(1,int(self.numNodes/5))
            path3 = self.random_walk_from(end_node, length, allow_internal_cycles=False, max_internal_cycles=1)
            path3.append(cloud_node)
        else:
            path3 = self.get_path(end_node, cloud_node)
        if not path3:
            return 0, []  # No path from end node to cloud

        # Combine paths
        full_path = path1[:-1] + sequence_path + path3[1:]  # Exclude overlapping nodes

        # Calculate average probability
        avg_prob, num_links = self.get_path_probability(full_path)
        return full_path, avg_prob
    
    
    def query_cycle_limit(self, sequence, method='min', max_internal_cycle=1):
        """
        Find a cycle path from the cloud node to the cloud node that contains the node sequence,
        limiting the number of internal cycles.

        Parameters:
            sequence (list or str): The sequence of nodes to include in the path.
            method (str): The method to use for pathfinding ('min', 'max', or 'dfs').
            max_internal_cycle (int): The maximum number of internal cycles allowed in the path.

        Returns:
            tuple: A tuple containing the the path as a list of nodes and average probability 
        
        Example:
                >>> import IdealFlow.Network as net     # import package.module as alias
                >>> n = net.IFN()
                >>> adjList = {
                        'a': {'c': 1, 'd': 2},
                        'b': {'c': 1, 'e': 3, '#Z#': 10},
                        'c': {'e': 5},
                        'd': {'c': 5},
                        'e': {'a': 3, 'b': 5},
                        '#Z#': {'a': 10, 'b': 10}  # Cloud node connections
                    }
                >>> n.set_data(adjList)
                >>> path, probability = n.query_cycle_limit('ace', method='min', max_internal_cycle=0)
                >>> print(f"Query result: Probability: {probability}, Path: {path}")
                >>> path, probability = nquery_cycle_limit('ace', method='max', max_internal_cycle=1)
                >>> print(f"Query result with max_internal_cycle=1: Probability: {probability}, Path: {path}")
       
        """
        if isinstance(sequence, str):
            sequence = list(sequence)

        cloud_node = self.cloud_name  # The cloud node identifier

        # Step 1: Find path from cloud node to start node of the sequence
        start_node = None
        for node in sequence:
            if node in self.listNodes:
                start_node = node
                break
        if not start_node:
            return 0, []  # No nodes in sequence exist in the network

        # Initialize visited nodes and cycles
        visited_nodes = set()
        cycles = {'count': 0}

        # Path from cloud node to start_node
        path1 = self.find_path_cycle_limit(cloud_node, start_node, max_internal_cycle, visited_nodes, cycles)
        # path1 = self.backtracking_cycle_limit(cloud_node, start_node, max_internal_cycle, visited_nodes, cycles)
        if not path1:
            return 0, []  # No path from cloud to start node

        # Step 2: Build path through sequence
        sequence_path = []
        i = sequence.index(start_node)
        sequence_path.append(start_node)
        while i < len(sequence) - 1:
            node1 = sequence[i]
            node2 = sequence[i + 1]
            if node2 not in self.listNodes:
                i += 1
                continue  # Skip nodes not in network

            # Find path between node1 and node2
            path_seq = self.find_path_cycle_limit(node1, node2, max_internal_cycle, visited_nodes, cycles)
            # path_seq = self.backtracking_cycle_limit(node1, node2, max_internal_cycle, visited_nodes, cycles)
            if not path_seq:
                i += 1
                continue  # Skip if no path between nodes
            sequence_path.extend(path_seq[1:])  # Exclude first node to avoid duplication
            i += 1

        if not sequence_path:
            return 0, []  # No valid path through sequence

        # Step 3: Find path from end of sequence to cloud node
        end_node = None
        for node in reversed(sequence):
            if node in self.listNodes:
                end_node = node
                break
        if not end_node:
            return 0, []  # No nodes in sequence exist in the network

        path3 = self.find_path_cycle_limit(end_node, cloud_node, max_internal_cycle, visited_nodes, cycles)
        # path3 = self.backtracking_cycle_limit(end_node, cloud_node, max_internal_cycle, visited_nodes, cycles)
        if not path3:
            return 0, []  # No path from end node to cloud

        # Combine paths
        full_path = path1[:-1] + sequence_path + path3[1:]  # Exclude overlapping nodes

        # Calculate average probability
        avg_prob, num_links = self.get_path_probability(full_path)
        return full_path, avg_prob
    

    def is_path(self, trajectory: list) -> bool:
        """
        Check if the given trajectory is a valid path.

        Parameters:
            trajectory (list): A sequence of nodes.

        Returns:
            bool: True if the sequence forms a valid path, False otherwise.

        Example:
            >>> import IdealFlow.Network as net   
            >>> n = net.IFN()
            >>> n.add_link('a', 'b', 1)
            >>> n.add_link('b', 'c', 1)
            >>> n.is_path(['a', 'b', 'c'])
            True
        """
        if len(trajectory)<=1:
            return False
        for idx,node1 in enumerate(trajectory[:-1]):
            node2=trajectory[idx+1]
            weight=self.__getWeightLink__(node1,node2)
            if weight==0:
                return False
        return True
    
     
    def set_path(self, trajectory: list, delta_flow: float = 1) -> None:
        """
        Set a path in the network, updating the link weight by flow if it exists or creating the link with weight = 1 if it does not exist.

        Parameters:
            trajectory (list): A sequence of nodes to set as a path.
            delta_flow (float): The additional weight of the links alongthe path. Defaults to 1.

        Returns:
            None

        Example:
            >>> import IdealFlow.Network as net 
            >>> n = net.IFN()
            >>> n.add_link('a', 'b', 1)
            >>> n.set_path(['a', 'b', 'c'], 3)
            >>> n.adjList
            {'a': {'b': 4}, 'b': {'c': 3}, 'c': {}}
        """
        for idx,startNode in enumerate(trajectory[:-1]):
            endNode=trajectory[idx+1]
            if delta_flow==1:
                self.set_link_weight_plus_1(startNode, endNode)
            else:
                self.add_link(startNode, endNode, delta_flow)

            

    def is_trajectory_cycle(self, path: list) -> bool:
        """
        Check if the given path forms a cycle (i.e., start and end nodes are the same).

        Parameters:
            path (list): A sequence of nodes.

        Returns:
            bool: True if the path forms a cycle, False otherwise.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> n.add_link('a', 'b', 1)
            >>> n.is_trajectory_cycle(['a', 'b', 'a'])
            False
            >>> n.add_link('b', 'a', 1)
            >>> n.is_trajectory_cycle(['a', 'b', 'a'])
            True
        """
        return self.is_path(path) and path[0]==path[-1]
    

    def cycle_length(self, cycle: list) -> int:
        """
        Return the number of edges in the cycle if it forms a valid cycle, otherwise return 0.

        Parameters:
            cycle (list): A sequence of nodes forming a cycle.

        Returns:
            int: The number of edges in the cycle, or 0 if not a cycle.

        Example:
            >>> import IdealFlow.Network as net  
            >>> n = net.IFN()
            >>> n.add_link('a', 'b', 1)
            >>> n.cycle_length(['a', 'b', 'a'])
            0
            >>> n.add_link('b', 'a', 1)
            >>> n.cycle_length(['a', 'b', 'a'])
            2
        """
        if self.is_trajectory_cycle(cycle):
            return len(cycle)-1
        else:
            return 0
     
    
    def cycle_sum_weight(self, cycle: list) -> float:
        """
        Return the sum of weights in the cycle if it forms a valid cycle, otherwise return 0.

        Parameters:
            cycle (list): A sequence of nodes forming a cycle.

        Returns:
            float: The sum of weights in the cycle, or 0 if not a cycle.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> n.add_link('a', 'b', 1)
            >>> n.add_link('b', 'a', 2)
            >>> n.cycle_sum_weight(['a', 'b', 'a'])
            3
        """
        if self.is_trajectory_cycle(cycle):
            return self.path_sum_weight(cycle)
        else:
            return 0
        
    
    def path_length(self,startNode: str, endNode: str) -> int:
        """
        Return the number of edges in the path if it forms a valid path, otherwise return 0.

        Parameters:
            startNode (str): The starting node.
            endNode (str): The target node.

        Returns:
            int: The number of edges in the shortest path, or 0 if not a path.

        Example:
            >>> import IdealFlow.Network as net 
            >>> n = net.IFN()
            >>> n.add_link('a', 'b', 3)
            >>> n.add_link('b', 'c', 7)
            >>> n.path_length('a', 'c')
            2
            >>> n.path_length('c', 'a')
            0
        """
        shortest=self.shortest_path(startNode,endNode)
        if self.is_path(shortest):
            return len(shortest)-1
        else:
            return 0
    
    
    def path_distance(self, startNode: str, endNode: str) -> float:
        """
        Calculate the total weight of the shortest path between startNode and endNode.

        Parameters:
            startNode (str): The starting node.
            endNode (str): The target node.

        Returns:
            float: The sum of weights along the shortest path.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> n.add_link('a', 'b', 3)
            >>> n.add_link('b', 'c', 7)
            >>> n.path_distance('a', 'c')
            10
            >>> n.path_distance('c', 'a')
            0
        """
        shortest=self.shortest_path(startNode,endNode)
        return self.path_sum_weight(shortest)
    
    
    def path_sum_weight(self, path: list) -> float:
        """
        Return the sum of weights along the given path.

        Parameters:
            path (list): A sequence of nodes forming a path.

        Returns:
            float: The sum of weights along the path.

        Example:
            >>> import IdealFlow.Network as net  
            >>> n = net.IFN()
            >>> n.add_link('a', 'b', 1)
            >>> n.add_link('b', 'c', 2)
            >>> n.path_sum_weight(['a', 'b', 'c'])
            3
        """
        sum=0
        for idx,node1 in enumerate(path[:-1]):
            node2=path[idx+1]
            weight=self.__getWeightLink__(node1,node2)
            sum=sum+weight
        return sum    
    
    def random_walk_from(self, startNode: str, length: int = 1, allow_internal_cycles: bool = True, max_internal_cycles: int = float('inf')) -> list:
        """
        Perform a stochastic random walk from the given startNode, either until reaching a sink node 
        (a node with no outgoing edges) or for a specified number of steps. The next node is chosen 
        probabilistically based on the weights of the outgoing edges from the current node.

        Internal Cycle Handling:
            If `allow_internal_cycles` is False, the walk will terminate upon revisiting any node.
            If `allow_internal_cycles` is True, the walk can revisit nodes, but the number of internal cycles is limited by `max_internal_cycles`.
            
        Parameters:
            startNode (str): The starting node.
            length (int): The maximum number of steps in the random walk.
            allow_internal_cycles (bool): Whether to allow internal cycles (revisiting nodes) during the walk. Defaults to True.
            max_internal_cycles (int): The maximum number of internal cycles allowed. Defaults to infinity.

        Returns:
            list: A list of nodes visited during the random walk. The walk will stop if it reaches a sink node, if the specified length is reached, or if the number of allowed internal cycles is exceeded.

        Stochastic Behavior:
            If the current node has multiple outgoing neighbors, the next node is chosen based on the probability distribution proportional to the weights of the edges.
            Nodes with higher edge weights are more likely to be chosen than those with lower weights.
            
        Example: 
            >>> import IdealFlow.Network as net     # import package.module as alias
            >>> n = net.IFN()
            >>> n.add_link('a', 'b', 1)
            >>> n.add_link('b', 'c', 2)
            >>> n.add_link('b', 'd', 2)
            >>> n.add_link('d', 'a', 2)
            >>> n.add_link('d', 'c', 2)
            >>> print(n.random_walk_from('a', 7))
            >>> # n.show() 
            ['a', 'b', 'd', 'a', 'b', 'c']
            >>> print(n.random_walk_from('a', 7))
            ['a', 'b', 'd', 'a', 'b', 'd', 'a']
            >>> print(n.random_walk_from('a', 7))
            ['a', 'b', 'd', 'c']
            >>> print(n.random_walk_from('a', 7, allow_internal_cycles=False))
            ['a', 'b', 'c']
            >>> print(n.random_walk_from('a', 7, max_internal_cycles=1))
            ['a', 'b', 'd', 'a', 'b']
        """
        result = []
        currentNode = startNode
        result.append(currentNode)
        visitedNodes = set()  # Track visited nodes
        internalCycleCount = 0  # Track the number of internal cycles
    
        visitedNodes.add(currentNode)

        for n in range(length-1):
            toNodes = self.out_neighbors(currentNode)
            if toNodes:  # Check if current node has outgoing neighbors
                listNodes, listWeight = zip(*toNodes.items())
                # Stochastic choice: Probability proportional to edge weights
                probs = [x / sum(listWeight) for x in listWeight]
                currentNode = np.random.choice(listNodes, p=probs)
                # Check for internal cycle
                if currentNode in visitedNodes:
                    if not allow_internal_cycles:
                        break  # If internal cycles are not allowed, stop the walk
                    else:
                        internalCycleCount += 1
                        if internalCycleCount > max_internal_cycles:
                            break  # Stop the walk if internal cycles exceed the allowed number
                
                result.append(currentNode)
                visitedNodes.add(currentNode)
            else:  # Reached a sink node, stop the walk
                break
        
        return result

    
    def random_cycle_from(self, startEndNode: str, allow_internal_cycles: bool = True, max_internal_cycles: int = float('inf')) -> list:
        """
        Perform a random walk that starts and ends at the specified startEndNode, forming a cycle.

        Parameters:
            startEndNode (str): The starting and ending node of the cycle.
            allow_internal_cycles (bool, optional): If True (default), the walk allows internal cycles (cycles from nodes other than startEndNode).
            max_internal_cycles (int, optional): Maximum number of allowed internal cycles. Defaults to infinity if not specified.

        Returns:
            list: A list of nodes visited during the random walk that forms a cycle. If no cycle can be formed, it returns an empty list.
        
        Stochastic Behavior:
            The next node in the walk is chosen probabilistically based on the weights of the outgoing edges.
            Nodes with higher edge weights have a greater chance of being chosen, making the walk biased toward those edges.
        
        Special Cases:
            If the network contains no cycles, the method will return an empty list, as no cycle can be formed.
            If only one cycle exists in the network, the walk will discover that cycle.
            In strongly connected networks, the random walk is more likely to find cycles, as all nodes are reachable from any other node.

        Example: 
            >>> import IdealFlow.Network as net     # import package.module as alias
            >>> n = net.IFN()
            >>> n.add_link('a', 'b', 1)
            >>> n.add_link('b', 'c', 2)
            >>> n.add_link('b', 'd', 2)
            >>> n.add_link('d', 'a', 2)
            >>> n.add_link('d', 'c', 2)
            >>> print(n.random_cycle_from('a'))
            ['a', 'b', 'd', 'a']
            []
            >>> n.add_link('c', 'a', 2)    
            >>> print(n.random_cycle_from('a', allow_internal_cycles=False)) # Disallowing Internal Cycles
            ['a', 'b', 'd', 'c', 'a']
            >>> print(n.random_cycle_from('a', allow_internal_cycles=True, max_internal_cycles=2)) # Allowing Internal Cycles with a Limit
            ['a', 'b', 'c', 'a']
            >>> print(n.random_cycle_from('a', allow_internal_cycles=True)) # Unrestricted Internal Cycles 
            
        """
        result = []
        currentNode = startEndNode
        visitedNodes = set()  # Track visited nodes
        internalCycleCount = 0  # Track number of internal cycles
        
        # Initialize the result with the start node
        result.append(currentNode)
        visitedNodes.add(currentNode)
        
        while True:
            toNodes = self.out_neighbors(currentNode)
            if toNodes:  # There are outgoing edges
                listNodes, listWeight = zip(*toNodes.items())
                probs = [x / sum(listWeight) for x in listWeight]
                currentNode = np.random.choice(listNodes, p=probs)
                result.append(currentNode)
                
                if currentNode == startEndNode:
                    # We've returned to the start node, forming a cycle
                    break
                
                # Check for internal cycles
                if currentNode in visitedNodes:
                    if currentNode == startEndNode:
                        # This case is already handled (returning to start node)
                        break
                    elif allow_internal_cycles:
                        internalCycleCount += 1
                        if internalCycleCount > max_internal_cycles:
                            # Exceeded the allowed number of internal cycles
                            result = []
                            break
                    else:
                        # If internal cycles are not allowed, terminate the walk
                        result = []
                        break
                
                visitedNodes.add(currentNode)
            else:
                # No outgoing edges from this node, so no cycle is possible
                result = []
                break
        
        return result
    
    
    def get_path_probability(self, trajectory: list, isUpdateFirst: bool = False) -> tuple:
        """
        Compute the average probability of traversing the given trajectory (path) and the number of links.

        The traverse will end when it reaches the first zero flow link.
        If trajectory has no path, avg prob = 0.
        If the network probability was not computed, it will update the probabilities before calculation.

        Parameters:
            trajectory (list): A sequence of nodes forming a path.
            isUpdateFirst (bool): Whether to update the network probability before calculation. Defaults to False.

        Returns:
            tuple: A tuple containing:
                - avg_prob (float): The average probability of traversing the path.
                - num_links (int): The number of links traversed in the path.

        Example:
            >>> import IdealFlow.Network as net 
            >>> n = net.IFN()
            >>> n.add_link('a', 'b', 1)
            >>> n.add_link('b', 'c', 1)
            >>> n.get_path_probability(['a', 'b', 'c'])
            (0.5, 2)
        """
        # return self.__calculate_path_stat(trajectory, isUpdateFirst, calc_type="probability")
        if trajectory==[]:
            return 0,0
        if self.network_prob=={} or isUpdateFirst==True:
            self.__updateNetworkProbability__() # update self.network_prob
        sumProb=0
        numLink=0
        for idx,startNode in enumerate(trajectory[:-1]):
            endNode=trajectory[idx+1]
            adjList=self.network_prob
            prob=self.__getLinkWeight__(startNode,endNode,adjList)
            numLink=numLink+1
            if prob==0:
                sumProb=0
                break
            sumProb=sumProb+prob
        if numLink>0:
            avgProb=sumProb/numLink
        else:
            avgProb=0
        return avgProb,numLink
    
    
    def get_path_entropy(self, trajectory: list, isUpdateFirst: bool = False) -> float:
        """
        Calculate the entropy of a given trajectory (path).

        If the network probability was not computed, it will update the probabilities before calculation.
        The link probability must be greater than zero to be included in entropy computation.

        Parameters:
            trajectory (list): A sequence of nodes forming a path.
            isUpdateFirst (bool): Whether to update the network probability before calculation. Defaults to False.

        Returns:
            float: The entropy of the path.

        Example:
            >>> import IdealFlow.Network as net 
            >>> n = net.IFN()
            >>> n.add_link('a', 'b', 1)
            >>> n.add_link('b', 'c', 1)
            >>> n.get_path_entropy(['a', 'b', 'c'])
            0.9182958340544896
        """
        # return self.__calculate_path_stat(trajectory, isUpdateFirst, calc_type="entropy")
        if trajectory==[]:
            return 0
        if self.network_prob=={} or isUpdateFirst==True:
            self.__updateNetworkProbability__() # update self.network_prob
        sumEntropy=0
        numLink=0
        for idx,startNode in enumerate(trajectory[:-1]):
            endNode=trajectory[idx+1]
            adjList=self.network_prob#.adjList
            linkProb=self.__getLinkWeight__(startNode,endNode,adjList)
            numLink=numLink+1
            if linkProb>0:
                sumEntropy=sumEntropy-linkProb*math.log(linkProb,2)
        if numLink>0:
            avgEntropy=sumEntropy/numLink
        else:
            avgEntropy=0
        return avgEntropy
    
    def __calculate_path_stat(self, trajectory: list, isUpdateFirst: bool, calc_type: str) -> tuple:
        """
        Internal method to calculate path statistics (either probability or entropy).

        Parameters:
            trajectory (list): A sequence of nodes forming a path.
            isUpdateFirst (bool): Whether to update the network probability before calculation.
            calc_type (str): The type of calculation to perform ('probability' or 'entropy').

        Returns:
            tuple: For probability, returns (average probability, number of links).
                   For entropy, returns the total entropy value.
        """
        if not trajectory:
            return (0, 0) if calc_type == "probability" else 0

        if self.network_prob == {} or isUpdateFirst:
            self.__updateNetworkProbability__()

        sum_stat = 0
        num_links = 0

        for idx, start_node in enumerate(trajectory[:-1]):
            end_node = trajectory[idx + 1]
            link_prob = self.__getLinkWeight__(start_node, end_node, self.network_prob)
            num_links += 1

            if link_prob == 0:
                return (0, num_links) if calc_type == "probability" else 0

            if calc_type == "probability":
                sum_stat += link_prob
            elif calc_type == "entropy" and link_prob > 0:
                sum_stat -= link_prob * math.log(link_prob, 2)

        avg_stat = sum_stat / num_links if num_links > 0 else 0
        return (avg_stat, num_links) if calc_type == "probability" else avg_stat

    
    '''
    
        SEARCH METHODS
        
    '''
    

    def dfs(self, startNode: str) -> list:
        """
        Perform Depth-First Search (DFS) traversal starting from startNode.
        This method explores the graph by visiting a node, then recursively visiting its unvisited neighbors.
        The traversal continues until all reachable nodes from startNode are visited. It doesn't look for a
        specific destination.
        
        Use this method when you want to visit all connected nodes. It will visit nodes until all have been explored (order may vary based on graph structure).
        
        Parameters:
            startNode (str): The starting node.

        Returns:
            list: A list of nodes visited during DFS traversal.

        Example:
            >>> import IdealFlow.Network as net 
            >>> n = net.IFN()
            >>> n.add_link('a', 'b')
            >>> n.add_link('b', 'c')
            >>> n.add_link('a', 'd')
            >>> n.dfs('a')
            ['a', 'd', 'b', 'c']
        """
        vertices=self.nodes
        dfsPath=[]
        visited = dict.fromkeys(vertices,False)
        
        # Create DFS stack
        stack = [] 
        stack.append(startNode)  # push startNode
  
        while (len(stack)):  
            currentNode = stack.pop()           # pop a node from stack   
            if (not visited[currentNode]):  
                dfsPath.append(currentNode)
                visited[currentNode] = True 
  
                # Get all adjacent nodes
                # if not been visited, then push to stack  
                toNodes=self.out_neighbors(currentNode)
                for node in toNodes:
                    if (not visited[node]):  
                        stack.append(node)
        return dfsPath

    
    def dfs_until(self, startNode: str, endNode: str) -> list:
        """
        Perform Depth-First Search (DFS) traversal starting from startNode until endNode is reached.

        This method behaves like DFS but halts as soon as the specified endNode is encountered. It's useful
        when you want to find a node but don't need to explore the entire graph.

        Use this method when you're looking for a node and want to stop traversal once you find it.

        Parameters:
            startNode (str): The starting node for the DFS traversal.
            endNode (str): The target node at which the traversal stops.

        Returns:
            list: A list of nodes visited during DFS traversal until the endNode is reached.

        Example:
            >>> import IdealFlow.Network as net 
            >>> n = net.IFN()
            >>> n.add_link('a', 'b', 1)
            >>> n.add_link('b', 'c', 1)
            >>> n.add_link('a', 'd', 1)
            >>> n.dfs_until('a', 'c')
            ['a', 'b', 'c']
        """
        dfsPath=[]
        vertices=self.nodes
        visited = dict.fromkeys(vertices,False)
  
        # Create DFS stack
        stack = [] 
        stack.append(startNode)  # push startNode
  
        while (len(stack)):  
            currentNode = stack.pop()           # pop a node from stack            
            if (not visited[currentNode]):  
                dfsPath.append(currentNode)
                visited[currentNode] = True 
                if currentNode==endNode:
                    return dfsPath
  
                # Get all adjacent nodes
                # if not been visited, then push to stack  
                toNodes=self.out_neighbors(currentNode)
                for node in toNodes:
                    
                    if (not visited[node]):  
                        stack.append(node)
        return dfsPath

    @staticmethod
    def dfs_adj_list(v, start, visited, stack, adj_list, cycles, nodes):
        """
        Perform DFS to find cycles in the adjacency list.
        """
        visited[v] = True
        stack.append(v)
        for w in adj_list[v]:
            if w == start and len(stack) >= 1:
                cycles.add(IFN.canonize(tuple([nodes[node] for node in stack])))
            elif not visited[w]:
                IFN.dfs_adj_list(w, start, visited, stack, adj_list, cycles, nodes)
        stack.pop()
        visited[v] = False
   

    def bfs(self, startNode: str) -> list:
        """
        Perform Breadth-First Search (BFS) traversal starting from startNode.

        Parameters:
            startNode (str): The starting node.

        Returns:
            list: A list of nodes visited during BFS traversal.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> n.add_link('a', 'b', 1)
            >>> n.add_link('b', 'c', 1)
            >>> n.bfs('a')
            ['a', 'b', 'c']
        """
        bfsPath=[]
        vertices=self.nodes
        visited = dict.fromkeys(vertices,False)
  
        # Create BFS queue
        queue = [] 
        queue.append(startNode)  # push startNode
        visited[startNode] = True 
  
        while (len(queue)):  
            currentNode = queue.pop(0)   # pop a node from queue   
            bfsPath.append(currentNode)
  
            # Get all adjacent nodes
            # if not been visited, then push to queue
            toNodes=self.out_neighbors(currentNode)
            for node in toNodes:
                if (not visited[node]):  
                    queue.append(node)
                    visited[node] = True 
        return bfsPath


    def bfs_until(self, startNode: str, endNode: str) -> list:
        """
        Perform Breadth-First Search (BFS) traversal starting from startNode until endNode is reached.

        Parameters:
            startNode (str): The starting node.
            endNode (str): The target node.

        Returns:
            list: A list of nodes visited during BFS traversal until endNode is reached.

        Example:
            >>> import IdealFlow.Network as net 
            >>> n = net.IFN()
            >>> n.add_link('a', 'b', 1)
            >>> n.add_link('b', 'c', 1)
            >>> n.bfs_until('a', 'c')
            ['a', 'b', 'c']
        """
        bfsPath=[]
        vertices=self.nodes
        visited = dict.fromkeys(vertices,False)
  
        # Create BFS queue
        queue = [] 
        queue.append(startNode)  # push startNode
        visited[startNode] = True 
  
        while (len(queue)):  
            currentNode = queue.pop(0)   # pop a node from queue   
            bfsPath.append(currentNode)
            if currentNode==endNode:
                return bfsPath
                    
            # Get all adjacent nodes
            # if not been visited, then push to queue
            toNodes=self.out_neighbors(currentNode)
            for node in toNodes:
                if (not visited[node]):  
                    queue.append(node)
                    visited[node] = True
                    
        return bfsPath
    

    def backtracking(self, startNode: str, endNode: str) -> list:
        """
        Perform DFS traversal with backtracking to find a path from startNode to endNode.

        This method is specifically designed to find a valid path between two nodes. It explores routes, and if
        it encounters a dead-end or an invalid path, it backtracks to explore other possible routes.

        Use this method when you want to find an exact path between two nodes, potentially with constraints.

        Parameters:
            startNode (str): The starting node of the path.
            endNode (str): The target node to reach.

        Returns:
            list: A list of nodes forming the valid path from startNode to endNode using backtracking.

        Example:
            >>> import IdealFlow.Network as net 
            >>> n = net.IFN()
            >>> n.add_link('a', 'b', 1)
            >>> n.add_link('b', 'c', 1)
            >>> n.add_link('a', 'd', 1)
            >>> n.backtracking('a', 'c')
            ['a', 'b', 'c']
        """
        dfsPath=[]
        vertices=self.nodes
        visited = dict.fromkeys(vertices,False)
        predecessor=dict.fromkeys(vertices)
  
        # Create DFS stack
        stack = [] 
        stack.append(startNode)  # push startNode
  
        while (len(stack)):  
            currentNode = stack.pop()           # pop a node from stack  
            
            if (not visited[currentNode]):  
                # if not been visited, then visit
                dfsPath.append(currentNode)
                visited[currentNode] = True 
                if currentNode==endNode:
                    return dfsPath
  
                # Get all adjacent nodes
                toNodes=self.out_neighbors(currentNode)
                if toNodes=={}:
                    # if no unvisited outneighbor then
                    # backtrack due to leaf node
                    while dfsPath!=[] and dfsPath[-1]==currentNode:
                        currentNode = predecessor[currentNode]
                        dfsPath.pop()
                        toNodes=self.out_neighbors(currentNode)
                        isFound=False
                        for node in toNodes:
                            if node in stack and (not visited[node]):
                                isFound=True
                                break
                        if isFound==True:
                            break
                else:
                    # if have unvisited outneighbor then push to stack  
                    isBacktrack=True
                    for node in toNodes:
                        if (not visited[node]):  
                            stack.append(node)
                            predecessor[node]=currentNode
                            isBacktrack=False
                    if isBacktrack==True:
                        # if no unvisited node
                        # should we backtrack or out?
                        # backtrack due to visited node
                        while dfsPath!=[] and dfsPath[-1]==currentNode:
                            currentNode = predecessor[currentNode]
                            dfsPath.pop()
                            toNodes=self.out_neighbors(currentNode)
                            isFound=False
                            for node in toNodes:
                                if node in stack and (not visited[node]):
                                    isFound=True
                                    break
                            if isFound==True:
                                break 
        return dfsPath
    

    def backtracking_cycle_limit(self, startNode, endNode, max_internal_cycle, visited_nodes, cycles, path=None):
        """
        Perform DFS traversal with backtracking to find a path from startNode to endNode,
        limiting the number of internal cycles.

        Parameters:
            startNode (str): The starting node.
            endNode (str): The target node.
            max_internal_cycle (int): The maximum number of internal cycles allowed.
            visited_nodes (set): The set of nodes already visited in the path.
            cycles (dict): A dictionary containing the current cycle count.
            path (list): The current path.

        Returns:
            list: A list of nodes forming the path from startNode to endNode.
        """
        if path is None:
            path = [startNode]
        else:
            path.append(startNode)

        if startNode == endNode:
            # Update the global visited nodes
            visited_nodes.update(path)
            return path

        if startNode not in self.adjList:
            path.pop()
            return None

        for neighbor in self.adjList[startNode]:
            if neighbor not in visited_nodes:
                result = self.backtracking_cycle_limit(neighbor, endNode, max_internal_cycle, visited_nodes, cycles, path)
                if result is not None:
                    return result
            else:
                if cycles['count'] < max_internal_cycle:
                    cycles['count'] += 1
                    result = self.backtracking_cycle_limit(neighbor, endNode, max_internal_cycle, visited_nodes, cycles, path)
                    if result is not None:
                        return result
        path.pop()
        return None

    '''
    
        DATA SCIENCE RELATED
        
    '''

    def unlearn(self,trajectory: list) -> None:
        """
        Unassign (subtract weight) a trajectory from the network by setting weight = -1 for each link along the trajectory.

        Parameters:
            trajectory (list): A list of nodes representing the trajectory (node sequence) to be unlearned.

        Example:
            >>> import IdealFlow.Network as net  
            >>> n = net.IFN()
            >>> n.add_link('a', 'b', 1)
            >>> n.add_link('b', 'c', 1)
            >>> n.unlearn(['a', 'b', 'c'])
        """
        for idx,startNode in enumerate(trajectory[:-1]):
            endNode=trajectory[idx+1]
            self.add_link(startNode, endNode, weight=-1)
            

    def assign(self, trajectory: list) -> None:
        """
        Assign a trajectory to the network by updating link weights. 
        
        Alias:
            `set_path(rajectory,1)`

        Parameters:
            trajectory (list): A list of nodes representing the trajectory (node sequence) to be assigned.

        Example:
            >>> import IdealFlow.Network as net 
            >>> n = net.IFN()
            >>> n.assign(['a', 'b', 'c'])
        """
        self.set_path(trajectory,1)
    
    
    def generate(self,startNode: str = None, is_cycle=True, allow_internal_cycles: bool = False, max_internal_cycles: int = float('inf')) -> list:
        """
        Generate a random walk or random cycle starting and ending at the specified cloud node.

        Internal Cycle Handling:
            If `allow_internal_cycles` is False, the walk will terminate upon revisiting any node.
            If `allow_internal_cycles` is True, the walk can revisit nodes, but the number of internal cycles is limited by `max_internal_cycles`.
            
        Parameters:
            startNode (str): The starting node from which the random cycle is generated. Default is self.cloud_name.
            is_cycle (bool): technique to generate whether random walk or random cycle. Default is True (i.e. random cycle)
            allow_internal_cycles (bool): Whether to allow internal cycles (revisiting nodes) during the walk. Defaults to False.
            max_internal_cycles (int): The maximum number of internal cycles allowed. Defaults to infinity.

        Returns:
            list: A list of nodes representing the random cycle.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> trajectory =  ['a','b','c','d','e','a']
            >>> n.assign(trajectory)
            >>> n.generate('b')
        """
        if startNode is None:
            startNode = self.cloud_name

        if is_cycle:
            return self.random_cycle_from(startNode, allow_internal_cycles, max_internal_cycles)
        else:
            length = np.random.randint(1,self.numNodes)
            return self.random_walk_from(startNode, length, allow_internal_cycles, max_internal_cycles)
    

    def match(self,trajectory: list, dicIFNs: dict) -> tuple:
        """
        Return the IFN from the dictionary with the maximum trajectory entropy and percentage of max entropy/sum of entropy.

        Parameters:
            trajectory (list): A list of nodes representing the trajectory.
            dicIFNs (dict): A dictionary where the keys are IFN names and values are IFN objects.

        Returns:
            tuple: The name of the IFN with the maximum entropy and the percentage (float).

        Example:
            >>> n1, n2 = IFN(), IFN()
            >>> n.match(['a', 'b'], {'IFN1': n1, 'IFN2': n2})
        """
        dicEntropy={}
        lst=[]
        for name,n in dicIFNs.items():
            # trajectory entropy of each IFN
            h=n.get_path_entropy(trajectory) 
            dicEntropy[name]=h
            lst.append(h)
        if dicEntropy:
            name=max(dicEntropy, key=dicEntropy.get)
            # idx=np.argmax(lstEntropy)
            # ma=np.max(lstEntropy)
            ma=dicEntropy[name]
            n1=dicIFNs[name]
            if sum(lst)>0:
                pctEntropy=ma/sum(lst)
            else:
                pctEntropy=0
            return n1.name,pctEntropy
        else:
            return "",0
            
    
    @staticmethod
    def trajectory_to_links(trajectory: list) -> list:
        """
        Given a list of nodes, generate a list of links (node pairs).
        Note: The trajectory is not necessarily a path in the network.

        Parameters:
            trajectory (list): A list of nodes.

        Returns:
            list: A list of links (node pairs).

        Example:
            >>> import IdealFlow.Network as net 
            >>> n = net.IFN()
            >>> n.trajectory_to_links(['a', 'b', 'c'])
            [['a', 'b'], ['b', 'c']]
            >>> n.trajectory_to_links(['a','b','c','d'])
             [['a','b'],['b','c'],['c','d']]
        """
        return [[trajectory[i], trajectory[i + 1]] for i in range(len(trajectory) - 1)]
        # lst=[]
        # for idx,startNode in enumerate(trajectory[:-1]):
        #     endNode=trajectory[idx+1]
        #     lst.append([startNode,endNode])
        # return lst
    
    
    @staticmethod
    def link_combination(trajectory: list) -> list:
        """
        Given a list of nodes, generate all one-way link combinations.
        Note: the trajectory is not necessarily a path in the network
        
        Parameters:
            trajectory (list): A list of nodes.

        Returns:
            list: A list of one-way link combinations.

        Example:
            >>> import IdealFlow.Network as net     # import package.module as alias
            >>> n = net.IFN()
            >>> n.link_combination(['a', 'b', 'c'])
            [['a', 'b'], ['a', 'c'], ['b', 'c']]
            >>> n.link_combination(['a','b','c','d'])
            [['a','b'],['a','c'],['a','d'],['b','c'],['b','d'],['c','d']]
        """
        return [[trajectory[i], trajectory[j]] for i in range(len(trajectory) - 1) for j in range(i + 1, len(trajectory))]
        # lst=[]
        # for idx,node1 in enumerate(trajectory[:-1]):
        #     for idx2,node2 in enumerate(trajectory[idx+1:]):
        #         lst.append([node1,node2])
        # return lst
    

    @staticmethod
    def link_permutation(trajectory: list) -> list:
        """
        Given a list of nodes, generate all two-way link permutations.
        Note: the trajectory is not necessarily a path in the network
        
        Parameters:
            trajectory (list): A list of nodes.

        Returns:
            list: A list of two-way link permutations.

        Example:
            >>> import IdealFlow.Network as net  
            >>> n = net.IFN()
            >>> n.link_permutation(['a', 'b', 'c'])
            [['a', 'b'], ['a', 'c'], ['b', 'a'], ['b', 'c'], ['c', 'a'], ['c', 'b']]
            >>> n.link_permutation(['a','b','c','d'])
            [['a','b'],['a','c'],['a','d'],['b','c'],['b','d'],['c','d'],
            ['b','a'],['c','a'],['d','a'],['c','b'],['d','b'],['d','c']]
        """
        return [[trajectory[i], trajectory[j]] for i in range(len(trajectory)) for j in range(len(trajectory)) if i != j]
        # lst=[]
        # for idx,node1 in enumerate(trajectory):
        #     for idx2,node2 in enumerate(trajectory):
        #         if len(trajectory)==1 or idx!=idx2:
        #             lst.append([node1,node2])
        # return lst
    
   
    @staticmethod
    def association_train(trajectory: list, net: dict) -> dict:
        """
        Train the IFN for association based on a trajectory by creating a complete graph and overlaying it onto the IFN.
        It assumes to have two ways link permutation.

        Parameters:
            trajectory (list): A list of nodes representing the trajectory.
            net (dict): The initial IFN (or an empty IFN).

        Returns:
            dict: The updated IFN.

        Example:
            >>> import IdealFlow.Network as net 
            >>> n = net.IFN()
            >>> n.association_train(['a', 'b', 'c'], {})
        """
        if not net:
            net = IFN()
        complete_graph = net.complete_graph(trajectory)
        net = net.overlay(complete_graph, net)
        return net
    
    
    @staticmethod
    def association_predict_trajectory(trajectory: list, net: dict) -> tuple:
        """
        Predict associations of itemset from a trajectory based on the network's direct links.

        Given a trajectory and IFN, predict the association of itemset based on direct link from trajectory complete graph to the IFN that is not in the complete graph.

        Parameters:
            trajectory (list): A list of nodes representing the trajectory.
            net (dict): The IFN.

        Returns:
            tuple: A dictionary of predictions (item -> flow), total support, and total confidence.

            prediction = sorted dictionary of item: flow
            supp=count of flow in trajectory items
            conf=count of flow in all direct links

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> n.association_predict_trajectory(['a', 'b'], {})
        """
        nodeFlow=net.nodes_flow
        supp=[]
        conf=0
        pred={}
        for startNode in trajectory:
            if startNode in net.adjList:
                supp.append(nodeFlow[startNode])
                toNodes=net.adjList[startNode]
                for endNode,flow in toNodes.items():
                    if endNode not in trajectory:
                        conf=conf+flow
                        if endNode not in pred:
                            pred[endNode]=flow
                        else:
                            pred[endNode]=pred[endNode]+flow
        prediction=dict(sorted(pred.items(), key=lambda item: item[1], reverse=True))
        return prediction, sum(supp), conf
    
    
    def association_predict_actor_net(self,netActor,netSystem):
        """
        """
        nodeFlow=netActor.nodes_flow
        supp=[]
        conf=0
        pred={}
        for startNode in netActor:
            if startNode in netSystem.adjList:
                supp.append(nodeFlow[startNode])
                toNodes=netSystem.adjList[startNode]
                for endNode,flow in toNodes.items():
                    if endNode not in netActor.adjList:
                        conf=conf+flow
                        if endNode not in pred:
                            pred[endNode]=flow
                        else:
                            pred[endNode]=pred[endNode]+flow
        prediction=dict(sorted(pred.items(), key=lambda item: item[1], reverse=True))
        return prediction, sum(supp), conf
        
    
    
    def order_markov_lower(self, trajSuper: list) -> list:
        """
        Convert a high-order Markov trajectory into a first-order Markov trajectory.

        Agreement: Separator between nodes in a supernode is '|'. Cloud node is '#z#' and always first-order.

        Parameters:
            trajSuper (list): A list of supernodes of K order representing the trajectory.

        Returns:
            list: A list representing the first-order Markov trajectory in hash code.

        Example:
            >>> import IdealFlow.Network as net  
            >>> n = net.IFN()
            >>> n.order_markov_lower(['a|b', 'b|c', '#z#'])
        """
        lstCloud=IFN.find_element_in_list(self.cloud_name, trajSuper)
        trajS=trajSuper
        try:
            for i in range(10):
                if self.cloud_name in trajS:
                    trajS.remove(self.cloud_name)
                
        except ValueError:
            pass

        delim='|'
        traj=[]
        for idx,superNode in enumerate(trajS):
            nodes=superNode.split(delim)
            if idx==len(trajS)-1:
                for nd in nodes:
                    traj.append(nd)
            else:
                traj.append(nodes[0])
        if len(lstCloud)==2:
            traj.insert(0,self.cloud_name)
            traj.insert(len(traj),self.cloud_name)
        return traj

    
    
    def order_markov_higher(self, trajectory: list, order: int) -> list:
        """
        Convert a first-order Markov trajectory into a higher-order Markov trajectory.
        
        Agreement: 
            separator between node in supernode is '|'
            cloud node is '#z#' and always first order

        Parameters:
            trajectory (list): A list of nodes representing the first-order Markov trajectory in hash code.
            order (int): The desired (higher) Markov order.

        Returns:
            list: A list of supernodes of K order representing the higher-order Markov trajectory.

        Example:
            >>> import IdealFlow.Network as net 
            >>> n = net.IFN()
            >>> n.order_markov_higher(['a', 'b', 'c'], 2)
        """
        delim='|'
        q=len(trajectory)
        lstCloud=IFN.find_element_in_list(self.cloud_name, trajectory)
        maxOrder=q-len(lstCloud)
        if order>maxOrder:
            order=maxOrder
        if order<1:
            order=1
        trajSuper=[]
        for ix in range(0,q-order+1):
            nextNodes=trajectory[ix:ix+order]
            superNode=''
            for idx,nd in enumerate(nextNodes):
                if idx==0:
                    superNode=superNode+nd
                else:
                    superNode=superNode+delim+nd
            if self.cloud_name in superNode:
                superNode=self.cloud_name
            trajSuper.append(superNode)
        return trajSuper
    

    
    def to_markov_order(self,trajectory: list, toOrder: int) -> list:
        """
        Convert a trajectory from any Markov order to a specified Markov order.

        Parameters:
            trajectory (list): A list of nodes representing the trajectory.
            toOrder (int): The desired Markov order.

        Returns:
            list: A list representing the trajectory in the specified Markov order.

        Example:
            >>> import IdealFlow.Network as net 
            >>> n = net.IFN()
            >>> n.to_markov_order(['a', 'b', 'c'], 2)
        """
        traj=self.order_markov_lower(trajectory) # put to first order first
        trajSuper=self.order_markov_higher(traj,toOrder) # before going higher markov order
        return trajSuper  
    


    '''
    
        TESTING NETWORK
        
    '''
    
    @staticmethod
    def is_equal_network(net1: 'IFN', net2: 'IFN') -> bool:
        """
        Check if two networks are equal by comparing their adjacency lists.

        Parameters:
            net1 (IFN): The first network.
            net2 (IFN): The second network.

        Returns:
            bool: True if both networks are equal, False otherwise.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> net1, net2 = n, n
            >>> n.is_equal_network(net1, net2)
            True
        """
        return net1.adjList==net2.adjList        
    
    
    def is_equivalent_ifn(self, ifn: 'IFN') -> bool:
        """
        Check if the current IFN is equivalent to another IFN based on the coefficient of variation of flow.

        Parameters:
            ifn (IFN): The IFN to compare.

        Returns:
            bool: True if both IFNs are equivalent, False otherwise.

        Example:
            >>> import IdealFlow.Network as net
            >>> n1 = net.IFN()
            >>> n2 = net.IFN()
            >>> n1.is_equivalent_ifn(n2)
            False
        """
        cov1=self.cov_flow
        cov2=ifn.cov_flow
        if abs(cov1-cov2)<self.epsilon:
            return True
        else:
            return False
    
    
    def is_reachable(self, startNode: str, endNode: str) -> bool:
        """
        Check if a node is reachable from another node using BFS.

        Parameters:
            startNode (str): The starting node.
            endNode (str): The node to check reachability for.

        Returns:
            bool: True if the endNode is reachable from startNode, False otherwise.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> n.is_reachable('A', 'B')
            True
        """
        vertices=self.nodes
        visited = [False for i in range(len(vertices))]  
  
        # we use BFS 
        queue = [] 
        queue.append(startNode)  # push startNode
        idx=vertices.index(startNode)
        visited[idx] = True 
  
        while (len(queue)):  
            startNode = queue.pop(0)   # pop a node from queue   
            if startNode==endNode:
                return True
  
            # Get all adjacent nodes
            # if not been visited, then push to queue
            toNodes=self.out_neighbors(startNode)            
            for node in toNodes:
                idx=vertices.index(node)
                if (not visited[idx]):  
                    queue.append(node)
                    visited[idx] = True 
        return False
    

    @property
    def is_contain_cycle(self) -> bool:
        """
        Check if the internal IFN contains a cycle using in-degree tracking.

        Returns:
            bool: True if a cycle is present, False otherwise.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> n.is_contain_cycle
            False
        """
        vertices=self.nodes
        n=len(vertices)
        in_degree=[0]*n # in_degrees of all nodes
      
        # Traverse adjacency lists to fill in_degrees of  
        # vertices. This step takes O(V+E) time 
        for startNode in vertices:
            toNodes=self.out_neighbors(startNode)
            for node in toNodes:
                idx=vertices.index(node)
                in_degree[idx]+=1
          
        # enqueue all vertices with in_degree 0 
        queue=[] 
        for i in range(len(in_degree)): 
            if in_degree[i]==0: 
                v=vertices[i]
                queue.append(v) 
          
        cnt=0 # Initialize count of visited vertices 
      
        # One by one dequeue vertices from queue and enqueue  
        # adjacents if in_degree of adjacent becomes 0  
        while(queue): 
      
            # Extract front of queue (or perform dequeue)  
            # and add it to topological order  
            nu=queue.pop(0) 
      
            # Iterate through all its neighbouring nodes  
            # of dequeued node u and decrease their in-degree by 1  
            if nu in self.adjList:
                toNodes=self.adjList[nu]
                for v in toNodes:
                    idx=vertices.index(v)
                    in_degree[idx]-=1
      
                    # If in-degree becomes zero, add it to queue 
                    idx=vertices.index(v)
                    if in_degree[idx]==0: 
                        queue.append(v) 
            cnt+=1
      
        # Check if there was a cycle  
        if cnt==n: 
            return False
        else: 
            return True
    
    @property
    def is_acyclic(self) -> bool:
        """
        Check if the internal network contains no cycle.

        Returns:
            bool: True if the network is acyclic, False if a cycle is present.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> n.is_acyclic
            True
        """
        return self.is_contain_cycle==False
    

    @property
    def is_connected(self) -> bool:
        """
        Check if the internal network is connected.

        Returns:
            bool: True if the network is connected, False otherwise.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> n.is_connected
            True
        """
        n=self.to_graph()      # make both directions
        vertices=n.nodes
        startNode=vertices[0]
        trajectory=n.dfs(startNode)
        if len(trajectory)!=len(vertices):
            return False
        # all nodes is visited in any direction
        return True
    

    @property
    def is_strongly_connected(self) -> bool:
        """
        Check if the internal network is strongly connected.

        Returns:
            bool: True if the network is strongly connected, False otherwise.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> n.is_strongly_connected
            False
        """
        vertices=self.nodes
        
        # If BFS doesn't visit all nodes it is not strongly connected
        for startNode in vertices:
            trajectory=self.bfs(startNode)
            if len(trajectory)!=len(vertices):
                return False
            
        # reverse the links
        net=self.reverse_network()

        # If BFS of reverse network doesn't visit all nodes it is not strongly connected
        for startNode in vertices:
            trajectory=net.bfs(startNode)
            if len(trajectory)!=len(vertices):
                return False
        
        return True # otherwise strongly connected
    

    @property
    def is_premagic(self) -> bool:
        """
        Check if the in-weight and out-weight of all nodes are approximately equal.

        Returns:
            bool: True if the in-weight is approximately equal to out-weight for all nodes, False otherwise.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> n.is_premagic
            True
        """
        inWeigh,vertices=self.in_weight
        outWeigh,vertices=self.out_weight
        for i,v in enumerate(vertices):
            if abs(inWeigh[i]-outWeigh[i])>self.epsilon:
                return False
        return True
    

    @property
    def is_ideal_flow(self) -> bool:
        """
        Check if the network is an ideal flow network (premagic and strongly connected).

        Returns:
            bool: True if the network is an ideal flow, False otherwise.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> n.is_ideal_flow
            True
        """
        return self.is_strongly_connected and self.is_premagic
        

    @property
    def is_eulerian_cycle(self) -> bool:
        """
        Check if the internal network contains an Eulerian cycle.

        Returns:
            bool: True if the network contains an Eulerian cycle, False otherwise.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> n.is_eulerian_cycle
            False
        """
        # Check if all non-zero degree vertices are connected 
        if not self.is_strongly_connected:
            return False
  
        # Check if in degree and out degree of every vertex is same 
        inDeg,vertices=self.in_degree
        outDeg,vertices=self.out_degree
        for i,v in enumerate(vertices):
            if inDeg[i]!=outDeg[i]:
                return False
  
        return True
    

    @property
    def is_bipartite(self) -> bool:
        """
        Check if the internal network is bipartite.

        Returns:
            bool: True if the network is bipartite, False otherwise.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> n.is_bipartite
            True
        """
        M,nodes=self.get_matrix()
        V=self.total_nodes
        color = [-1] * V  
              
        #start is vertex 0  
        pos = 0 
        # two colors 1 and 0  
        retVal=self.color_graph(M, color, pos, 1)
#        if retVal: print('color=',color,'\n')
        return retVal
    
    

    '''
        NETWORK SET-LIKE OPERATIONS
        
    '''
    @staticmethod
    def union(net1: 'IFN', net2: 'IFN') -> 'IFN':
        """
        Return the union of two networks, combining links and nodes from both.

        Parameters:
            net1 (IFN): The first network.
            net2 (IFN): The second network.

        Returns:
            IFN: A new network containing the union of net1 and net2.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> net1, net2 = net.IFN(), net.IFN()
            >>> union_net = n.union(net1, net2)
        """
        n = IFN()
        for startNode1, toNodes1 in net1.adjList.items():
            for endNode1, weight1 in toNodes1.items():
                n.add_link(startNode1, endNode1, weight1)
        for startNode2, toNodes2 in net2.adjList.items():
            for endNode2, weight2 in toNodes2.items():
                n.add_link(startNode2, endNode2, weight2)

        # Subtract the intersection from the union
        intersect_net = IFN.intersect(net1, net2)
        return n.difference(n, intersect_net)
        # n=IFN()  # create new network
        
        # for startNode1 in net1.adjList:
        #     toNodes1=net1.adjList[startNode1]
        #     for endNode1,weight1 in toNodes1.items():
        #         n.add_link(startNode1,endNode1,weight1)
        # for startNode2 in net2.adjList:        
        #     toNodes2=net2.adjList[startNode2]                
        #     for endNode2,weight2 in toNodes2.items():
        #         n.add_link(startNode2,endNode2,weight2)
        # n1=IFN()
        # n1=n1.intersect(net1, net2)
        # n1=n.difference(n, n1)            
        # return n1
    

    @staticmethod
    def overlay(net1: 'IFN', net2: 'IFN') -> 'IFN':
        """
        Overlay net1 into net2, updating weights in net2 based on net1.

        Parameters:
            net1 (IFN): The network to overlay. (smaller)
            net2 (IFN): The base network to overlay onto. (base - usually larger)

        Returns:
            IFN: The updated network with the overlay of net1.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> base_net, overlay_net = net.IFN(), net.IFN()
            >>> result_net = n.overlay(base_net, overlay_net)
        """
        for startNode, toNodes in net1.adjList.items():
            for endNode,weight in toNodes.items():
                net2.add_link(startNode,endNode,weight)
        return net2
        

    @staticmethod
    def difference(net2: 'IFN', net1: 'IFN') -> 'IFN':
        """
        Subtract the link flow of net1 from net2.

        Reduce link flow of net2 based on net1 = set difference (net2-net1)

        Parameters:
            net1 (IFN): The network to subtract. (smaller)
            net2 (IFN): The base network to subtract from.(base - usually larger)

        Returns:
            IFN: The updated network after the subtraction.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> base_net, diff_net = net.IFN(), net.IFN()
            >>> result_net = n.difference(base_net, diff_net)
        """
        n = net2.duplicate()
        for startNode, toNodes in net1.adjList.items():
            for endNode, flow in toNodes.items():
                n.add_link(startNode, endNode, weight=-flow)
        return n
    

    @staticmethod
    def complement(net: 'IFN') -> 'IFN':
        """
        Return the complement of the given network (complete graph minus the network).

        Parameters:
            net (IFN): The network to complement.

        Returns:
            IFN: The complement of the network.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> result_net = n.complement(n)
        """
        n = IFN()
        U = n.universe(net)
        return n.difference(U, net)
    

    @staticmethod
    def intersect(net1: 'IFN', net2: 'IFN') -> 'IFN':
        """
        Return the intersection of two networks.

        Parameters:
            net1 (IFN): The first network.
            net2 (IFN): The second network.

        Returns:
            IFN: A new network representing the intersection of net1 and net2.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> net1, net2 = net.IFN(), net.IFN()
            >>> intersect_net = n.intersect(net1, net2)
        """
        n=IFN()  # create new network
        for startNode in net1.adjList:
            if startNode in net2.adjList:   # if startNode exists in net2
                toNodes=net1.adjList[startNode]
                toNodes2=net2.adjList[startNode]
                for endNode1,weight1 in toNodes.items():
                    for endNode2,weight2 in toNodes2.items():
                        if endNode1==endNode2:
                            weight=min(weight1,weight2)
                            n.add_link(startNode,endNode1,weight)
        return n


    @staticmethod
    def universe(net: 'IFN') -> 'IFN':
        """
        Return the universe (complete digraph) of the given network.

        Parameters:
            net (IFN): The network to create a universe from.

        Returns:
            IFN: A complete graph representing the universe of the network.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> universe_net = n.universe(n)
        """
        vertices=net.nodes
        m=net.max_flow
        U=net.complete_graph(vertices,weight=m)
        return U
    

    '''
        NEW NETWORKS
        
    '''
    
    @staticmethod
    def complete_graph(trajectory: list, weight: int = 1) -> 'IFN':
        """
        Create a complete graph from a trajectory list, with two-way links.
        If trajectory has only one item, create a node.
        A complete graph weight 1 is always an IFN.

        Parameters:
            trajectory (list): List of nodes.
            weight (int): The weight of each link. Default is 1.

        Returns:
            IFN: A new complete graph.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> trajectory = ['A', 'B', 'C']
            >>> complete_net = n.complete_graph(trajectory)
        """
        n = IFN()
        if len(trajectory) == 1:
            n.add_node(trajectory[0])
        else:
            for startNode in trajectory:
                for endNode in trajectory:
                    if startNode != endNode:
                        n.add_link(startNode, endNode, weight)
        return n
        # n=IFN()  # create new network
        # for idx1,startNode in enumerate(trajectory):
        #     for idx2,endNode in enumerate(trajectory):
        #         if len(trajectory)==1:
        #             n.add_node(trajectory[0])
        #         elif idx1!=idx2:
        #             n.add_link(startNode, endNode,weight)
        # return n
    
    
    def duplicate(self) -> 'IFN':
        """
        Create a duplicate of the current network.
        (not the same reference)

        Returns:
            IFN: A new network that is a deep copy of the current one.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> duplicate_net = n.duplicate()
        """
        n=IFN()  # create new network
        # main copy
        n.name=self.name
        n.adjList=copy.deepcopy(self.adjList)
        # also copy
        n.listNodes=copy.deepcopy(self.listNodes)
        n.numNodes=self.numNodes        
        return n
    
    
    def to_graph(self) -> 'IFN':
        """
        Convert the digraph to a graph, making the adjacency matrix symmetric.
        The link weights are adjusted to all 1.
        
        Returns:
            IFN: A new network representing the graph counterpart of the digraph.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> graph_net = n.to_graph()
        """
        n=IFN()  # create new network
        nR=self.reverse_network()
        
        # copy from current
        for startNode in self.adjList.keys(): 
            toNodes=self.out_neighbors(startNode)
            for endNode,weight in toNodes.items():
                n.add_link(endNode,startNode,1)
        # copy also from reverse
        for startNode in nR.adjList.keys(): 
            toNodes=nR.out_neighbors(startNode)
            for endNode,weight in toNodes.items():
                n.add_link(endNode,startNode,1)
        return n
    
    
    def reverse_network(self) -> 'IFN':
        """
        Return a new network with the direction of all links reversed.

        Returns:
            IFN: A new network with reversed links.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> reversed_net = n.reverse_network()
        """
        n=IFN()
        for startNode in self.adjList.keys(): 
            toNodes=self.out_neighbors(startNode)
            for endNode,weight in toNodes.items():
                n.add_link(endNode,startNode,weight)
        n.reindex()
        return n
    
    
    def network_delete_cloud(self) -> 'IFN':
        """
        Return a duplicate of the current network, with the cloud node removed.

        Returns:
            IFN: A new network without the cloud node.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> clean_net = n.network_delete_cloud()
        """
        net=self.duplicate()
        net.delete_node(self.cloud_name)
        return net
    


    '''
    
        MATRICES
        
    '''

    def get_matrix(self) -> tuple:
        """
        Return the adjacency matrix of the network and the list of nodes.

        Returns:
            tuple: A tuple containing the adjacency matrix and list of nodes.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> adj_matrix, nodes = n.get_matrix()
        """
        return self.__adjList2Matrix__(self.adjList)
    
    
    def set_matrix(self, M: list, listNode: list = []) -> None:
        """
        Replace the adjacency list with the provided matrix.
        This is useful if we use matrices in computation and
        want to put the matrix into network

        Parameters:
            M (list): Adjacency matrix to set.
            listNode (list): List of node names. Defaults to generating Excel-like labels.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> matrix = [[0, 1], [1, 0]]
            >>> n.set_matrix(matrix)
        """
        if listNode==[]:
            # set up default list node if not specified
            size=np.array(M).shape
            mC=size[0]
            listNode=[IFN.num_to_excel_col(x) for x in range(1,mC+1)]
        self.adjList=self.__matrix2AdjList__(np.array(M),listNode)
        self.listNodes=listNode
        self.numNodes=len(self.listNodes)                  # replace the numNodes

    
    @staticmethod
    def binarized_matrix(M: list) -> list:
        """
        Convert the matrix to a binary (0, 1) matrix.

        Parameters:
            M (list): The input matrix.

        Returns:
            list: The binarized (0, 1) version of the matrix.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> matrix = [[0, 2], [1, 0]]
            >>> bin_matrix = n.binarized_matrix(matrix)
            >>> print(bin_matrix)
            [[0, 1], [1, 0]]
        """
        return [[int(bool(x)) for x in l] for l in M]

    '''
    
        CORE IFN
        
    '''
    

    @staticmethod
    def ideal_flow_to_stochastic(F: np.ndarray) -> np.ndarray:
        """
        Convert an ideal flow matrix into a Markov stochastic matrix.

        Parameters:
            F (np.ndarray): The ideal flow matrix.

        Returns:
            np.ndarray: The stochastic matrix.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> F = np.array([[0.2, 0.3], [0.4, 0.1]])
            >>> S = n.ideal_flow_to_stochastic(F)
            >>> print(S)
            [[0.4 0.6]
             [0.8 0.2]]
        """
        s=np.apply_along_axis(np.sum, axis=1, arr=F)
        return F/s[:,np.newaxis]

    
    @staticmethod
    def markov(S: np.ndarray, kappa: float = 1) -> np.ndarray:
        """
        Compute the steady-state Markov vector from a stochastic matrix.
        Exact computation approach.

        Parameters:
            S (np.ndarray): The stochastic matrix.
            kappa (float): Total of the Markov vector. Default is 1.

        Returns:
            np.ndarray: The Markov chain.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> S = np.array([[0.5, 0.5], [0.5, 0.5]])
            >>> pi = n.markov(S)
            >>> print(pi)
            [[0.5]
             [0.5]]
        
        Previous version: steadyStateMC()

        See also:
            :meth:`stochastic_to_pi()`

            :meth:`ideal_flow()`
        """
       
        if not isinstance(S, np.ndarray):
            S = np.array(S)
        [m,n]=S.shape
        if m==n:
            I=np.eye(n)
            j=np.ones((1,n))
            X=np.concatenate((np.subtract(S.T,I), j), axis=0) # vstack
            try:
                Xp = np.linalg.pinv(X)  # Moore-Penrose inverse
            except np.linalg.LinAlgError:
                # If SVD does not converge, return a uniform distribution as a fallback
                print("Warning: SVD did not converge. Returning uniform distribution as a fallback.")
                return np.ones((n, 1)) * (kappa / n)
    
            y=np.zeros((m+1,1),float)
            y[m]=kappa
            pi=np.dot(Xp,y)
            return pi
        else:
            raise ValueError("Input matrix S must be square.")


    @staticmethod
    def ideal_flow(S: np.ndarray, pi: np.ndarray) -> np.ndarray:
        """
        Compute the ideal flow matrix from a stochastic matrix and Perron vector.

        Parameters:
            S (np.ndarray): The stochastic matrix.
            pi (np.ndarray): The Perron vector.

        Returns:
            np.ndarray: The ideal flow matrix.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> S = np.array([[0.5, 0.5], [0.5, 0.5]])
            >>> pi = np.array([[0.5], [0.5]])
            >>> F = n.ideal_flow(S, pi)
            >>> print(F)
            [[0.25 0.25]
             [0.25 0.25]]
        """
        return S*pi
        
        

    @staticmethod
    def adjacency_to_ideal_flow(A: np.ndarray, kappa: float = 1) -> np.ndarray:
        """
        Convert an adjacency matrix into an ideal flow matrix of equal distribution of outflow.

        Parameters:
            A (np.ndarray): The adjacency matrix.
            kappa (float): The total flow. Default is 1.

        Returns:
            np.ndarray: The ideal flow matrix.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> A = np.array([[0, 1], [1, 0]])
            >>> F = n.adjacency_to_ideal_flow(A)
            >>> print(F)
            [[0.5 0.5]
             [0.5 0.5]]
        """
        S=IFN.adjacency_to_stochastic(A)
        pi=IFN.markov(S,kappa)
        return IFN.ideal_flow(S,pi)
        

    @staticmethod
    def capacity_to_ideal_flow(C: np.ndarray, kappa: float = 1) -> np.ndarray:
        """
        Convert a capacity matrix into an ideal flow matrix.

        Parameters:
            C (np.ndarray): The capacity matrix.
            kappa (float): The total flow. Default is 1.

        Returns:
            np.ndarray: The ideal flow matrix.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> C = np.array([[0, 2], [3, 0]])
            >>> F = n.capacity_to_ideal_flow(C)
            >>> print(F)
            [[0.4 0.6]
             [0.5 0.5]]
        """
        S = IFN.capacity_to_stochastic(C)
        pi = IFN.markov(S, kappa)
        return IFN.ideal_flow(S, pi)
    

    @staticmethod
    def congestion(F: np.ndarray, C: np.ndarray) -> np.ndarray:
        """
        Compute the congestion matrix from flow and capacity matrices.
        congestion matrix is element wise division of flow/capacity, except zero remain zero
        
        Parameters:
            F (np.ndarray): The flow matrix.
            C (np.ndarray): The capacity matrix.

        Returns:
            np.ndarray: The congestion matrix.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> F = np.array([[1, 2], [2, 1]])
            >>> C = np.array([[2, 2], [2, 2]])
            >>> congestion_matrix = n.congestion(F, C)
            >>> print(congestion_matrix)
            [[0.5 1. ]
             [1.  0.5]]
        """
        return IFN.hadamard_division(F,C)
    

    @staticmethod
    def is_square_matrix(M: np.ndarray) -> bool:
        """
        Check if a matrix is square.

        Parameters:
            M (np.ndarray): The input matrix.

        Returns:
            bool: True if the matrix is square, False otherwise.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> M = np.array([[1, 2], [3, 4]])
            >>> result = n.is_square_matrix(M)
            >>> print(result)
            True
        """
        return M.shape[0] == M.shape[1]

    
    @staticmethod
    def is_non_negative_matrix(M: np.ndarray) -> bool:
        """
        Check if all elements in a matrix are non-negative.

        Parameters:
            M (np.ndarray): The input matrix.

        Returns:
            bool: True if all elements are non-negative, otherwise False.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> M = np.array([[0, 1], [2, 3]])
            >>> result = n.is_non_negative_matrix(M)
            >>> print(result)
            True
        """
        return np.all(np.array(M) >= 0)

    
    @staticmethod
    def is_positive_matrix(M: np.ndarray) -> bool:
        """
        Check if all elements in a matrix are positive.

        Parameters:
            M (np.ndarray): The input matrix.

        Returns:
            bool: True if all elements are positive, otherwise False.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> M = np.array([[1, 2], [3, 4]])
            >>> result = n.is_positive_matrix(M)
            >>> print(result)
            True
        """
        return np.all(np.array(M) > 0)


    @staticmethod
    def is_premagic_matrix(M: np.ndarray) -> bool:
        """
        Check if a matrix is premagic (row sums equal column sums).

        Parameters:
            M (np.ndarray): The input matrix.

        Returns:
            bool: True if the matrix is premagic, otherwise False.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> M = np.array([[1, 2], [2, 1]])
            >>> result = n.is_premagic_matrix(M)
            >>> print(result)
            True
        """
        if not IFN.is_square_matrix(np.array(M)):
            return False
        row_sums = np.sum(M, axis=1)
        col_sums = np.sum(M, axis=0)
        return np.allclose(row_sums, col_sums)
        
    
    @staticmethod
    def is_irreducible_matrix(M: np.ndarray) -> bool:
        """
        Check if a matrix is irreducible.
        This method is slow for large matrix.

        Parameters:
            M (np.ndarray): The input matrix.

        Returns:
            bool: True if the matrix is irreducible, otherwise False.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> M = np.array([[0, 1], [1, 0]])
            >>> result = n.is_irreducible_matrix(M)
            >>> print(result)
            True
        """
        M=np.array(M)
        if IFN.is_square_matrix(M) and IFN.is_non_negative_matrix(M):
            [m,n]=M.shape
            I=np.eye(n)
            Q=np.linalg.matrix_power(np.add(I,M),n-1) # Q=(I+M)^(n-1)
            return IFN.is_positive_matrix(Q)
        else:
            return False
    

    @staticmethod
    def is_ideal_flow_matrix(mA: np.ndarray) -> bool:
        """
        Check if a matrix is an ideal flow matrix.

        Parameters:
            mA (np.ndarray): The input matrix.

        Returns:
            bool: True if the matrix is an ideal flow matrix, otherwise False.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> M = np.array([[0.5, 0.5], [0.5, 0.5]])
            >>> result = n.is_ideal_flow_matrix(M)
            >>> print(result)
            True
        """
        mA=np.array(mA)
        return IFN.is_premagic_matrix(mA) and IFN.is_irreducible_matrix(mA)    
   

    @staticmethod
    def equivalent_ifn(F: np.ndarray, scaling: float, is_rounded: bool = False) -> np.ndarray:
        """
        Compute an equivalent ideal flow network with the given scaling.

        Parameters:
            F (np.ndarray): The flow matrix.
            scaling (float): The scaling factor.
            is_rounded (bool): Whether to round the result. Default is False.

        Returns:
            np.ndarray: The equivalent ideal flow matrix.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> F = np.array([[0.2, 0.3], [0.4, 0.1]])
            >>> scaled_F = n.equivalent_ifn(F, scaling=2)
            >>> print(scaled_F)
            [[0.4 0.6]
             [0.8 0.2]]
        """
        F = np.array(F)
        if is_rounded:
            return np.round(F * scaling)
        return (F * scaling)


    @staticmethod
    def global_scaling(F: np.ndarray, scaling_type: str = 'min', val: float = 1) -> float:
        """
        Compute a global scaling factor for a flow matrix.
        Return scaling factor to ideal flow matrix to get equivalent IFN

        Parameters:
            F (np.ndarray): The ideal flow matrix.
            scaling_type (str): The type of scaling ('min', 'max', 'sum', 'int', 'avg', 'std', 'cov'). 'int' means basis IFN (minimum integer)
            val (float): The value for scaling. Default is 1.

        Returns:
            float: The scaling factor.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> F = np.array([[0.2, 0.3], [0.4, 0.1]])
            >>> scaling = n.global_scaling(F, 'min')
            >>> print(scaling)
            5.0
        """
        f=np.ravel(F[np.nonzero(F)]) # list of non-zero values in F
        if len(f) == 0:
            return 1
        if scaling_type == 'min':
            return val / np.min(f)
        elif scaling_type == 'max':
            return val / np.max(f)
        elif scaling_type == 'sum':
            return val / np.sum(f)
        elif scaling_type == 'int':
            fractions = [IFN.decimal_to_fraction(x) for x in f]
             # Use only positive denominators for LCM calculation
            denominators = np.array([abs(frac[1]) for frac in fractions], dtype=np.int64)            
            common_denominator = np.lcm.reduce(denominators)
            # common_denominator = np.lcm.reduce([frac[1] for frac in fractions])
            return common_denominator
        elif scaling_type == 'avg':
            return val / np.mean(f)
        elif scaling_type == 'std':
            return val / np.std(f)
        elif scaling_type == 'cov':
            mean = np.mean(f)
            std = np.std(f)
            return val / (std / mean)
        else:
            raise ValueError("unknown scalingType")
        

    @staticmethod
    def min_irreducible(k: int) -> np.ndarray:
        """
        Generate the minimum irreducible matrix of size k.

        Parameters:
            k (int): The size of the matrix.

        Returns:
            np.ndarray: The minimum irreducible matrix.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> A = n.min_irreducible(3)
            >>> print(A)
            [[0 1 0]
             [0 0 1]
             [1 0 0]]
        """
        A=np.zeros((k,k), dtype=int)
        for r in range(k-1):
            c=r+1
            A[r,c]=1
        A[k-1,0]=1
        return A


    
    @staticmethod
    def add_random_ones(A: np.ndarray, m: int = 6) -> np.ndarray:
        """
        Add random ones to a matrix until the total number of ones is equal to m.
        
        Add 1 to the matrix A at random cell location such that the total 1 in the matrix is equal to m. If total number of 1 is less than m, it will not be added.
        It will not add anything if the current number of ones is already larger than m

        Parameters:
            A (np.ndarray): The input square matrix.
            m (int): The number of ones to add. Default is 6.

        Returns:
            np.ndarray: The updated matrix.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> A = np.zeros((3, 3))
            >>> updated_A = n.add_random_ones(A, 4)
            >>> print(updated_A)
            [[0 1 0]
             [1 0 0]
             [0 1 1]]
        """
        A = np.array(A)
        n = A.shape[0]  # assuming square matrix
        current_ones = np.sum(A)  # count current number of 1s

        if current_ones < m:  # only add if needed
            needed_ones = m - current_ones  # number of 1s to add
            available_positions = np.argwhere(A == 0)  # get all positions where A is 0

            if len(available_positions) < needed_ones:
                raise ValueError("Not enough empty spaces to add more ones.")

            # Randomly select positions to add ones
            chosen_positions = available_positions[np.random.choice(len(available_positions), needed_ones, replace=False)]

            for pos in chosen_positions:
                A[pos[0], pos[1]] = 1  # add a 1 at the chosen position

        return A
        # n = len(A)
        # n2 = np.sum(A)   # total number of 1 in the matrix
        # if m>n2:         # only add 1 if necessary
        #     k=0          # k is counter of additional 1
        #     for g in range(n*n):                 # repeat until max (N by N) all filled with 1
        #         idx=np.random.randint(0, n*n-1)  # create random index
        #         row=math.ceil(idx/n)         # get row of the random index
        #         col=((idx-1)%n)+1            # get col of the random index
        #         if A[row-1,col-1]==0:        # only fill cell that still zero
        #             A[row-1,col-1]=1
        #             k=k+1
        #             if k==m-n2:              # if number of links M has been reached
        #                 break                # break from for-loop of g (before reaching max)
        # return A

    

    @staticmethod
    def rand_irreducible(num_nodes: int = 5, num_links: int = 8) -> np.ndarray:
        """
        Generate a random irreducible matrix.

        Parameters:
            num_nodes (int): Number of nodes.
            num_links (int): Number of links.

        Returns:
            np.ndarray: The random irreducible matrix.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> A = n.rand_irreducible(3, 4)
            >>> print(A)
            [[0 1 0]
             [1 0 1]
             [1 1 0]]
        """
        A = IFN.min_irreducible(num_nodes)
        A1 = IFN.add_random_ones(A, num_links)
        P = IFN.rand_permutation_eye(num_nodes)
        A2 = P @ A1 @ P.T
        return A2
    

    @staticmethod
    def rand_permutation_eye(n: int = 5) -> np.ndarray:
        """
        Generate a random permutation of the identity matrix.

        Parameters:
            n (int): The size of the matrix.

        Returns:
            np.ndarray: The permuted identity matrix.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> permuted_eye = n.rand_permutation_eye(3)
            >>> print(permuted_eye)
            [[0. 1. 0.]
             [1. 0. 0.]
             [0. 0. 1.]]
        """
        eye = np.eye(n, dtype=int)
        np.random.shuffle(eye)
        return eye


        
    
    '''
    
        UTILITIES
    
    '''
    
    
    def reindex(self) -> None:
        """
        Sort the nodes and standardize the adjacency list.

        This operation may take a long time for large datasets.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> n.reindex()
        """
        m,listNode=self.__adjList2Matrix__(self.adjList)
        self.adjList=self.__matrix2AdjList__(m,listNode)
    
    
    def save(self, fileName: str) -> None:
        """
        Save the adjacency list to a JSON file.

        Parameters:
            fileName (str): The name of the file to save the data.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> n.save('network.json')
        """
        with open(fileName,'w') as fp:
            json.dump(self.adjList,fp,sort_keys=True,indent=4)
    
    
    def load(self, fileName: str) -> None:
        """
        Load the adjacency list from a JSON file.

        Parameters:
            fileName (str): The name of the file to load the data from.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> n.load('network.json')
        """
        with open(fileName,'r') as fp:
            self.adjList=json.load(fp)
    

    @staticmethod
    def read_csv(fName: str) -> list:
        """
        Read a CSV file and return a 2D array.

        Parameters:
            fName (str): The name of the CSV file to read.

        Returns:
            list: A 2D array representing the content of the CSV file.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> data = n.read_csv('data.csv')
            >>> print(data)
            [['1', '2'], ['3', '4']]
        """  
        data=[]
        with open(fName, newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                col=list(map(str.strip,row))
                data.append(col)
        return data
    

    @staticmethod
    def __flatten__(nDArray: np.ndarray) -> list:
        """
        Flatten an n-dimensional numpy array into a 1D list.

        Parameters:
            nDArray (np.ndarray): The input n-dimensional array.

        Returns:
            list: A __flatten__ed 1D list.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> arr = np.array([[1, 2], [3, 4]])
            >>> flat_list = n.__flatten__(arr)
            >>> print(flat_list)
            [1, 2, 3, 4]
        """
        return list(np.concatenate(nDArray).flat) 
    
    
    @staticmethod
    def __lcm__(a: int, b: int) -> int:
        """
        Compute the least common multiple (LCM) of two integers.

        Parameters:
            a (int): The first integer.
            b (int): The second integer.

        Returns:
            int: The least common multiple of a and b.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> result = n.__lcm__(12, 15)
            >>> print(result)
            60
        """
        return a*b // math.gcd(a,b)
    
    @staticmethod
    def __lcm_list__(lst: list) -> int:
        """
        Compute the least common multiple (LCM) of a list of integers.

        Parameters:
            lst (list): A list of integers.

        Returns:
            int: The least common multiple of all integers in the list.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> result = n.__lcm_list__([3, 4, 5])
            >>> print(result)
            60
        """
        a = lst[0]
        for b in lst[1:]:
            a = IFN.__lcm__(a,b)
        return a


    @staticmethod
    def hadamard_division(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Perform elementwise division of two matrices, ignoring division by zero.

        Parameters:
            a (np.ndarray): The numerator matrix.
            b (np.ndarray): The denominator matrix.

        Returns:
            np.ndarray: The result of the elementwise division. Zero is returned for divisions by zero.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> A = np.array([[1, 2], [3, 0]])
            >>> B = np.array([[2, 2], [0, 2]])
            >>> result = n.hadamard_division(A, B)
            >>> print(result)
            [[0.5 1. ]
             [0.  0. ]]
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            c = np.true_divide( a, b )
            c[ ~ np.isfinite( c )] = 0  # -inf inf NaN
        return c
    
    
    '''
    
    Show the Network
    
    '''
    
    
    def show(self, layout: str = 'Planar', mNode: list = None, arrThreshold: list = None, routes: list = None) -> nx.DiGraph:
        """
        Visualizes the network using matplotlib and NetworkX.

        Parameters:
            layout (str, optional): The layout of the graph visualization. Options include:
                'Bipartite', 'Circular', 'Fruchterman', 'Kawai', 'Planar', 'Random', 
                'Shell', 'Spectral', 'Spiral', 'Spring'. Defaults to 'Planar'.
            mNode (list, optional): Custom positions for nodes, each row should be [node_id, x, y].
            arrThreshold (list, optional): Thresholds for edge weights to determine the edge color.
                Format: [low_threshold, high_threshold].
            routes (list, optional): A list of routes, where each route is a list of node IDs. 
                Edges in these routes are highlighted in red.

        Returns:
            nx.DiGraph: The directed graph representing the IFN.

        Example:
            >>> import IdealFlow.Network as net
            >>> adjList = {'a': {'b': 1.5}, 'b': {'c': 0.7}, 'c': {}}
            >>> n = net.IFN(adjList)
            >>> n.show(layout='Planar')
        """
        try:
            vertices = self.nodes
            totalNodes = len(vertices)

            plt.figure()
            G = nx.DiGraph()
            G.add_nodes_from(vertices)

            for n1, dic in self.adjList.items():
                for n2, cost in dic.items():
                    G.add_edge(n1, n2, weight=int(round(cost, 2)))

            if mNode is None:
                pos = self._get_layout(G, layout)
            else:
                pos = self._get_custom_node_positions(G, mNode)

            self._draw_edges(G, pos, arrThreshold, totalNodes)
            if routes is not None:
                self._highlight_routes(G, pos, routes)

            nodes = nx.draw_networkx_nodes(G, pos, node_color='w')
            nodes.set_edgecolor('black')
            nx.draw_networkx_labels(G, pos, font_size=14, font_family='sans-serif')

            plt.axis('off')
            plt.title(f"{self.name} ($\\kappa$={round(self.total_flow, 2)})", fontsize=26, fontweight=15)
            plt.show()

            return G

        except Exception as e:
            raise RuntimeError(f"An error occurred while displaying the network: {e}")
           

    def _get_layout(self, G: nx.Graph, layout: str) -> dict:
        """
        Determines the layout of the network visualization.

        Parameters:
            G (nx.Graph): The graph to layout.
            layout (str): The desired layout.

        Returns:
            dict: A dictionary of node positions.

        Raises:
            ValueError: If an unsupported layout is specified.

        Example:
            >>> G = nx.DiGraph()
            >>> G.add_nodes_from([1, 2, 3])
            >>> pos = IFN()._get_layout(G, 'Random')
        """       
        if layout == "Bipartite":
            top = nx.bipartite.sets(G)[0]
            return nx.bipartite_layout(G, top)
        elif layout == "Circular":
            return nx.circular_layout(G)
        elif layout == "Circular":
            return nx.circular_layout(G)
        elif layout == "Fruchterman":
            return nx.fruchterman_reingold_layout(G)
        elif layout == "Kawai":
            return nx.kamada_kawai_layout(G)
        elif layout == "Planar":
            return nx.planar_layout(G)   
        elif layout == "Random":
            return nx.random_layout(G)
        elif layout == "Shell":
            return nx.shell_layout(G)
        elif layout =="Spectral":
            return nx.spectral_layout(G)
        elif layout =="Spiral":
            return nx.spiral_layout(G)
        elif layout=="Spring":
            return nx.spring_layout(G)        
        else:
            return nx.circular_layout(G)  # Default layout


    def _get_custom_node_positions(self, G: nx.Graph, mNode: list) -> dict:
        """
        Sets custom positions for the nodes.

        Parameters:
            G (nx.Graph): The graph object.
            mNode (list): A list of custom positions [node_id, x, y].

        Returns:
            dict: A dictionary of node positions.
        
        Example:
            >>> G = nx.DiGraph()
            >>> mNode = [[1, 0.1, 0.5], [2, 0.4, 0.8]]
            >>> pos = IFN()._get_custom_node_positions(G, mNode)
        """
        try:
            pos = {}
            for node in mNode:
                node_id, x, y = node
                G.add_node(node_id, pos=(x, y))
                pos[node_id] = (x, y)
            return pos
        except Exception as e:
            raise ValueError(f"Error setting custom node positions: {e}")

    def _get_edge_color(self, weight: float, arrThreshold: list) -> str:
        """
        Determines the color of an edge based on its weight and thresholds.

        Parameters:
            weight (float): The weight of the edge.
            arrThreshold (list): A list of thresholds [low, high].

        Returns:
            str: The color of the edge.
        
        Example:
            >>> IFN()._get_edge_color(1.5, [1, 2])
            'yellow'
        """
        if arrThreshold:
            if weight < arrThreshold[0]:
                return 'green'
            elif weight < arrThreshold[1]:
                return 'yellow'
            return 'red'
        return 'black'

    def _draw_edges(self, G: nx.Graph, pos: dict, arrThreshold: list, totalNodes: int) -> None:
        """
        Draws edges on the graph.

        Parameters:
            G (nx.Graph): The graph.
            pos (dict): Node positions.
            arrThreshold (list, optional): Thresholds for edge colors.
            totalNodes (int): The total number of nodes.

        Example:
            >>> G = nx.DiGraph()
            >>> G.add_edge(1, 2, weight=2)
            >>> pos = {1: (0, 0), 2: (1, 1)}
            >>> IFN()._draw_edges(G, pos, [1, 3], 2)
        """
        totalWeight = G.size(weight='weight')
        for (node1, node2, data) in G.edges(data=True):
            weight = data['weight']
            width = (weight * totalNodes / totalWeight * 5) if totalWeight > 0 else 1
            color = self._get_edge_color(weight, arrThreshold)
            if G.has_edge(node2, node1):
                self.__draw_curve__(pos, node1, node2, weight, width, color)
            else:
                self.__draw_arrow__(pos, node1, node2, weight, width, color)

    def _highlight_routes(self, G: nx.Graph, pos: dict, routes: list) -> None:
        """
        Highlights the specified routes on the graph.

        Parameters:
            G (nx.Graph): The graph.
            pos (dict): Node positions.
            routes (list): A list of routes to highlight.

        Example:
            >>> G = nx.DiGraph()
            >>> pos = {1: (0, 0), 2: (1, 1), 3: (2, 2)}
            >>> routes = [[1, 2, 3]]
            >>> IFN()._highlight_routes(G, pos, routes)
        """
        edges = [(route[i], route[i + 1]) for route in routes for i in range(len(route) - 1)]
        nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color='red', width=1.2)


    @staticmethod
    def __draw_curve__(pos, node1, node2, weight, width, color):
        """
        Draws a curved arrow between two nodes.

        Parameters:
            pos (dict): Node positions.
            node1 (str): The starting node.
            node2 (str): The ending node.
            weight (float): The weight of the edge.
            width (float): The width of the arrow.
            color (str): The color of the arrow.

        Example:
            >>> pos = {1: (0, 0), 2: (1, 1)}
            >>> IFN().__draw_curve__(pos, 1, 2, 1.5, 2, 'red')
        """
        ax = plt.gca()
        ax.annotate("", xy=pos[node2], xycoords='data',
                    xytext=pos[node1], textcoords='data',
                    arrowprops=dict(width=0.01, shrink=0.1, color=color, linewidth=width,
                                    connectionstyle="arc3,rad=-0.25"))
        shift = (-0.1, 0)
        plt.annotate(text=weight, xy=(0, 0), xytext=(0.6 * np.array(pos[node2]) + 0.4 * np.array(pos[node1])) + shift,
                     color='red', size=16, textcoords='data')

    @staticmethod
    def __draw_arrow__(pos, node1, node2, weight, width, color):
        """
        Draws a straight arrow between two nodes.

        Parameters:
            pos (dict): Node positions.
            node1 (str): The starting node.
            node2 (str): The ending node.
            weight (float): The weight of the edge.
            width (float): The width of the arrow.
            color (str): The color of the arrow.

        Example:
            >>> pos = {1: (0, 0), 2: (1, 1)}
            >>> IFN().__draw_arrow__(pos, 1, 2, 1.5, 2, 'black')
        """
        ax = plt.gca()
        ax.annotate("", xy=pos[node2], xycoords='data',
                    xytext=pos[node1], textcoords='data',
                    arrowprops=dict(width=0.01, shrink=0.045, color=color, linewidth=width,
                                    connectionstyle="arc3,rad=-0.001"))
        shift = (-0.08, -0.025)
        plt.annotate(text=weight, xy=(0, 0), xytext=(0.56 * np.array(pos[node2]) + 0.44 * np.array(pos[node1])) + shift,
                     color='red', size=16, textcoords='data')
    


    @staticmethod
    def to_adjacency_matrix(matrix):
        '''
        Convert a non-negative matrix to a binary (0, 1) adjacency matrix.
        
        Parameters:
            matrix (list of list of int/float): The input matrix.

        Returns:
            np.array: The adjacency matrix.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> matrix = [[0, 1], [1, 0]]
            >>> n.to_adjacency_matrix(matrix)
            array([[0, 1], [1, 0]])
        '''
        try:
            A = np.array(matrix) > 0
            return A.astype(int)
        except Exception as e:
            raise ValueError(f"Error converting to adjacency matrix: {e}")
        
    @staticmethod
    def capacity_to_adjacency(matrix):
        '''
        Convert a non-negative matrix to a binary (0, 1) adjacency matrix.
        
        Parameters:
            matrix (list of list of int/float): The input matrix.

        Returns:
            np.array: The adjacency matrix.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> C = [[1, 2], [3, 4]]
            >>> n.capacity_to_adjacency(C)

        Alias of :meth:`to_adjacency_matrix()`

        See also: 
            :meth:`to_adjacency_matrix()`
        '''
        return IFN.to_adjacency_matrix(matrix)

    @staticmethod
    def capacity_to_stochastic(C, method='row', alpha=1, beta=0.00001, epsilon=1e-10):
        '''
        Convert a capacity matrix into a stochastic matrix using various methods.

        Parameters:
            C (list of list of int/float or 2D np.array): The capacity matrix.
            method (str): Method for conversion: 'row', 'col', 'alpha_beta_row', 'alpha_beta_col'. 
            Options are:
                - 'row': Converts to a row stochastic matrix.
                - 'col': Converts to a column stochastic matrix.
                - 'alpha_beta_row': Uses the alpha and beta parameters for row-wise transformation.
                - 'alpha_beta_col': Uses the alpha and beta parameters for column-wise transformation.
            alpha (float): The alpha parameter, used only for 'alpha_beta' and 'alpha_beta_col' methods.
            beta (float): The beta parameter, used only for 'alpha_beta' and 'alpha_beta_col' methods.
            epsilon (float): A small value to avoid division by zero, used in 'row' and 'col' methods.

        Returns:
            np.array: The resulting stochastic matrix.
        
        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> C = [[1, 2], [3, 4]]
            >>> n.capacity_to_stochastic(C, method='row')
            
        '''
        try:
            C = np.array(C)
            if C.shape[0] != C.shape[1]:
                return "Capacity Matrix must be a square matrix."
            
            if method == 'row':
                row_sum = np.sum(C, axis=1)
                row_sum = np.where(row_sum == 0, epsilon, row_sum)  # Avoid division by zero
                return C / row_sum[:, np.newaxis]

            elif method == 'col':
                col_sum = np.sum(C, axis=0)
                col_sum = np.where(col_sum == 0, epsilon, col_sum)  # Avoid division by zero
                return C / col_sum

            elif method == 'alpha_beta_row':
                C_alpha = np.power(C, alpha)
                C_beta = np.exp(beta * C)
                C_transformed = C_alpha * C_beta
                denom = np.sum(C_transformed, axis=1, keepdims=True)
                denom = np.where(denom == 0, epsilon, denom)  # Avoid division by zero
                return C_transformed / denom

            elif method == 'alpha_beta_col':
                C_alpha = np.power(C, alpha)
                C_beta = np.exp(beta * C)
                C_transformed = C_alpha * C_beta
                denom = np.sum(C_transformed, axis=0, keepdims=True)
                denom = np.where(denom == 0, epsilon, denom)  # Avoid division by zero
                return C_transformed / denom

            else:
                raise ValueError("Invalid method. Choose 'row', 'col', 'alpha_beta_row', or 'alpha_beta_col'.")
        except Exception as e:
            raise ValueError(f"Error converting capacity to stochastic matrix: {e}")


    @staticmethod
    def to_equal_outflow(C):
        '''
        Return ideal flow matrix with equal outflow from capacity matrix.
        
        Parameters:
            C (list of list of int/float): The capacity matrix.

        Returns:
            list of list of int/float: The ideal flow matrix.
        
        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()        
            >>> C = [[1, 2], [0, 1]]
            >>> n.to_equal_outflow(C)
        '''
        try:
            A = IFN.to_adjacency_matrix(C)
            S = IFN.capacity_to_stochastic(A)
            F = IFN.stochastic_to_ideal_flow(S)
            scaling = IFN.global_scaling(F, 'int')
            return IFN.equivalent_ifn(F, scaling)
        except Exception as e:
            raise ValueError(f"Error generating equal outflow matrix: {e}")


    @staticmethod
    def to_equal_inflow(C):
        '''
        Return ideal flow matrix with equal inflow from capacity matrix.
        
        Parameters:
            C (list of list of int/float): The capacity matrix.

        Returns:
            list of list of int/float: The ideal flow matrix.
        
        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()        
            >>> C = [[1, 2], [0, 1]]
            >>> n.to_equal_inflow(C)
        '''
        try:
            A = np.transpose(IFN.to_adjacency_matrix(C))
            S = IFN.capacity_to_stochastic(A)
            F = np.transpose(IFN.stochastic_to_ideal_flow(S))
            scaling = IFN.global_scaling(F, 'int')
            return IFN.equivalent_ifn(F, scaling)
        except Exception as e:
            raise ValueError(f"Error generating equal inflow matrix: {e}")


    @staticmethod
    def capacity_to_balance_inflow_outflow(C, lambda_=0.5):
        '''
        Return ideal flow matrix balancing inflow and outflow from capacity matrix.
        
        Parameters:
            C (list of list of int/float): The capacity matrix.
            lambda_ (float): The lambda parameter to balance inflow and outflow.

        Returns:
            list of list of int/float: The balanced ideal flow matrix.
        
        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()        
            >>> C = [[1, 2], [0, 1]]
            >>> n.capacity_to_balance_inflow_outflow(C)
        '''
        try:
            F_in = IFN.to_equal_inflow(C)
            F_out = IFN.to_equal_outflow(C)
            F = (1 - lambda_) * np.array(F_in) + lambda_ * np.array(F_out)
            return F
        except Exception as e:
            raise ValueError(f"Error balancing inflow and outflow: {e}")


    @staticmethod
    def rand_capacity(num_node=5, max_capacity=9):
        """
        Generate random capacity matrix for a given number of nodes.
        
        Parameters:
            num_node (int): Number of nodes.
            max_capacity (int): Maximum capacity value.

        Returns:
            list of list of int: The random capacity matrix.
        
        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()        
            >>> n.rand_capacity(5, 10)
            [[0, 7, 3, 0, 0], ...]
        """
        try:
            num_link = np.random.randint(1, num_node // 2 + 1) * num_node + 1
            C = IFN.rand_irreducible(num_node, num_link)
            for i in range(num_node):
                C[i][i] = 0
            for i in range(num_node):
                for j in range(num_node):
                    if C[i][j] > 0:
                        C[i][j] = np.random.randint(1, max_capacity + 1)
            return C
        except Exception as e:
            raise ValueError(f"Error generating random capacity matrix: {e}")


    @staticmethod
    def random_irreducible_stochastic(N):
        """
        Generate random irreducible stochastic matrix.
        
        Parameters:
            N (int): Size of the matrix.

        Returns:
            list of list of float: The random stochastic matrix.
        
        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()        
            >>> n.random_irreducible_stochastic(3)
        """
        C = IFN.rand_capacity(N)
        Fstar = IFN.premier_ifn(C)
        denom = np.sum(Fstar, axis=1, keepdims=True)
        return (Fstar / denom)


    @staticmethod
    def random_ideal_flow_matrix(N: int, kappa: float = 1) -> np.array:
        """
        Generate random irreducible ideal flow matrix.

        Parameters:
            N (int): Size of the matrix (number of nodes).
            kappa (float): Scaling factor for the ideal flow. Default is 1.

        Returns:
            np.array: Random irreducible ideal flow matrix.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> result = n.random_ideal_flow_matrix(3, kappa=27)
            >>> print(result)
            [[...], [...], [...]]
        """
        try:
            C = IFN.rand_capacity(N)
            S = IFN.capacity_to_stochastic(C)
            return IFN.stochastic_to_ideal_flow(S, kappa)
        except Exception as e:
            raise ValueError(f"Error generating random ideal flow matrix: {e}")


    @staticmethod
    def __arr_sequence_to_markov__(arr: list) -> tuple:
        """
        Convert a sequence of numbers into a Markov transition matrix.

        Parameters:
            arr (list of int): Input sequence of integers representing transitions.

        Returns:
            tuple: A tuple containing:
                - Markov transition matrix (np.array)
                - Unique elements in the input array (np.array)

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> sequence = [1, 2, 2, 3, 1]
            >>> markov_matrix, unique_elements = n.__arr_sequence_to_markov__(sequence)
            >>> print(markov_matrix)
            [[...], [...], [...]]
            >>> print(unique_elements)
            [1 2 3]
        """
        try:
            unique, indices = np.unique(arr, return_inverse=True)
            M = np.zeros((len(unique), len(unique)))
            for i, j in zip(indices, indices[1:]):
                M[i, j] += 1
            return M, unique
        except Exception as e:
            raise ValueError(f"Error creating Markov matrix: {e}")


    @staticmethod
    def random_walk_matrix(m_capacity: np.array, arr_name: list, prev_index: int) -> tuple:
        """
        Perform a random walk on a Markov transition matrix.

        Parameters:
            m_capacity (np.array): The Markov transition matrix.
            arr_name (list of str): List of node names corresponding to the rows/columns of the matrix.
            prev_index (int): The index of the previous node in the walk.

        Returns:
            tuple: The next node name and its index.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> markov_matrix = np.array([[0, 1], [1, 0]])
            >>> arr_name = ['A', 'B']
            >>> next_node, next_index = n.random_walk_matrix(markov_matrix, arr_name, 0)
            >>> print(next_node, next_index)
            B 1
        """
        try:
            row = prev_index
            cnt = m_capacity[row]
            prob = np.cumsum(cnt / np.sum(cnt))
            r = np.random.rand()
            index = np.searchsorted(prob, r)
            return arr_name[index], index
        except Exception as e:
            raise ValueError(f"Error during random walk: {e}")

    
    def random_walk_nodes(self, start_node: str, length: int = 1) -> list:
        """
        Perform a random walk through nodes starting from a specific node.

        Parameters:
            start_node (str): The starting node for the random walk.
            length (int): The number of steps to walk. Default is 1.

        Returns:
            list: List of nodes visited during the random walk.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> result = n.random_walk_nodes('A', length=3)
            >>> print(result)
            ['A', 'B', 'C']
        """
        result = [start_node]
        current_node = start_node
        for _ in range(length):
            to_nodes = self.out_neighbors(current_node)
            if to_nodes:
                list_nodes = list(to_nodes.keys())
                list_weight = list(to_nodes.values())
                probs = [x / sum(list_weight) for x in list_weight]
                current_node = self.weighted_random_choice(list_nodes, probs)
                result.append(current_node)
        return result


    def random_walk_cycle(self, start_end_node: str) -> list:
        """
        Perform a random walk through nodes in a cycle starting and ending at the same node.

        Parameters:
            start_end_node (str): The node from which the random cycle starts and ends.

        Returns:
            list: List of nodes visited during the random walk cycle.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> result = n.random_walk_cycle('A')
            >>> print(result)
            ['A', 'B', 'C', 'A']
        """
        result = [start_end_node]
        current_node = start_end_node
        while True:
            to_nodes = self.out_neighbors(current_node)
            if to_nodes:
                list_nodes = list(to_nodes.keys())
                list_weight = list(to_nodes.values())
                probs = [x / sum(list_weight) for x in list_weight]
                current_node = self.weighted_random_choice(list_nodes, probs)
                result.append(current_node)
            if start_end_node == current_node:
                break
        return result


    @staticmethod
    def weighted_random_choice(list_nodes: list, probs: list) -> str:
        """
        Select a random node from a list based on given probabilities.

        Parameters:
            list_nodes (list of str): List of node names.
            probs (list of float): List of probabilities corresponding to each node.

        Returns:
            str: Selected node.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> nodes = ['A', 'B', 'C']
            >>> probabilities = [0.2, 0.5, 0.3]
            >>> result = n.weighted_random_choice(nodes, probabilities)
            >>> print(result)
            'B'
        """
        try:
            r = np.random.rand()
            cumulative = np.cumsum(probs)
            for i, prob in enumerate(cumulative):
                if r <= prob:
                    return list_nodes[i]
        except Exception as e:
            raise ValueError(f"Error in weighted random choice: {e}")


    @staticmethod
    def __matrix_round_integer__(matrix):
        """
        Round each element of the matrix to the nearest integer.
        
        Parameters:
            matrix (list of list of float): The input matrix.

        Returns:
            np.array: Rounded matrix.
        
        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()        
            >>> matrix = [[1.2, 2.5], [3.6, 4.1]]
            >>> n.__matrix_round_integer__(matrix)
            array([[1, 2], [4, 4]])
        """
        try:
            return np.round(matrix).astype(int)
        except Exception as e:
            raise ValueError(f"Error rounding matrix elements: {e}")


    @staticmethod
    def __is_list_larger_than__(list_, num=1):
        '''
        Check if all elements in the list are larger than a given number.
        
        Parameters:
            list_ (list of int/float): The input list.
            num (int/float): The number to compare against.

        Returns:
            bool: True if all elements are larger, otherwise False.
        
        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()        
            >>> n.__is_list_larger_than__([2, 3, 4], 1)
            True
        '''
        try:
            return all(x > num for x in list_)
        except Exception as e:
            raise ValueError(f"Error checking list values: {e}")


    @staticmethod
    def flows_in_cycle(F: list, cycle: tuple) -> list:
        """
        Return list of flows in a cycle.

        Parameters:
            F (list): The flow matrix.
            cycle (tuple or string): tuple of indices in a cycle or string cycle term

        Returns:
            list: The list of flows in the cycle.

        See also:
            :meth:`change_flow_in_cycle`

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> F = [[3, 1], 
                     [1, 0]]
            >>> adjL=n.flow_matrix_to_adj_list(F)
            >>> n.find_all_cycles_in_adj_list(adjL)
            [(0,), (0, 1)] 
            >>> cycle = (0, 1)
            >>> n.flows_in_cycle(F,cycle)
            [1, 1]
            >>> cycle = (0,)
            >>> n.flows_in_cycle(F,cycle)
            [3]
            >>> F=[[0, 3, 0],
                   [2, 0, 1],
                   [1, 0, 0]]
            >>> adjL=n.flow_matrix_to_adj_list(F)
            >>> n.find_all_cycles_in_adj_list(adjL)
            [(0, 1), (0, 1, 2)]
            >>> cycle = (0, 1)
            >>> n.flows_in_cycle(F,cycle)
            [3, 2]
            >>> cycle = (0, 1, 2)
            >>> n.flows_in_cycle(F,cycle)
            [3, 1, 1]
            >>> cycles=n.find_all_cycles_in_matrix(F)
            >>> print(cycles)
            [('a', 'b', 'c'), ('a', 'b')]
            >>> n.flows_in_cycle(F,cycle[0])
            [3, 1, 1]
            >>> n.flows_in_cycle(F,cycle[1])
            [3, 2]

        """
        list_flow = []
        for i in range(len(cycle)):
            if isinstance(cycle[i], str):
                row, col = IFN.__string_link_to_coord__(cycle[i] + cycle[(i + 1) % len(cycle)])
            else:            
                row = cycle[i]
                col = cycle[(i + 1) % len(cycle)]
            list_flow.append(F[row][col])
        return list_flow


    @staticmethod
    def change_flow_in_cycle(F: list, cycle: tuple, change: float = 1) -> list:
        """
        Add or subtract flow matrix based on the cycle.

        Parameters:
            F (list): The flow matrix.
            cycle (tuple or str): tuple of indices in a cycle or string cycle term
            change (float): The amount to change the flow. Default is 1.

        Returns:
            list: The updated flow matrix.

        See also:
            :meth:`flows_in_cycle`

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> F=[[0, 3, 0],
                   [2, 0, 1],
                   [1, 0, 0]]
            >>> adjL=n.flow_matrix_to_adj_list(F)
            >>> n.find_all_cycles_in_adj_list(adjL)
            [(0, 1), (0, 1, 2)]
            >>> cycle =(0, 1, 2)
            >>> F = n.change_flow_in_cycle(F, cycle, change=2)
            >>> print(F)
            [[0, 5, 0], 
             [2, 0, 3], 
             [3, 0, 0]]
            >>> cycle =(0, 1)
            >>> F = n.change_flow_in_cycle(F, cycle, change=2)
            print(F)
            [[0, 7, 0], 
             [4, 0, 3], 
             [3, 0, 0]]
        """
        for i in range(len(cycle)):
            if isinstance(cycle[i], str):
                row, col = IFN.__string_link_to_coord__(cycle[i] + cycle[(i + 1) % len(cycle)])
            else:
                row = cycle[i]
                col = cycle[(i + 1) % len(cycle)]
            F[row][col] += change
        return F


    @staticmethod
    def matrix_apply_cycle(flow_matrix: list, cycle: str, flow: float = 1) -> list:
        """
        Return the updated flow matrix after applying flow unit along the given cycle.

        Parameters:
            flow_matrix (list): The flow matrix.
            cycle (str): The cycle string.
            flow (float): The flow to apply. Default is 1.

        Returns:
            list: The updated flow matrix.

        Example:
            >>> import IdealFlow.Network as net
            >>> flow_matrix = [[0, 2], [3, 0]]
            >>> cycle = 'ab'
            >>> n = net.IFN()
            >>> updated_matrix = n.matrix_apply_cycle(flow_matrix, cycle, flow=2)
            >>> print(updated_matrix)
            [[0, 4], [5, 0]]
        """
        new_flow_matrix = np.array(flow_matrix, copy=True)
        for j in range(len(cycle)):
            from_node = ord(cycle[j]) - 97
            to_node = ord(cycle[(j + 1) % len(cycle)]) - 97
            new_flow_matrix[from_node, to_node] += flow
        return new_flow_matrix


    @staticmethod
    def __string_link_to_coord__(str_link: str) -> list:
        """
        Convert link string to coordinate.

        Parameters:
            str_link (str): The link string.

        Returns:
            list: The coordinates.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> n.__string_link_to_coord__('ab')            
            [0, 1]
            >>> n.__string_link_to_coord__('abcd')
            [0, 1, 2, 3] 
            >>> n.__string_link_to_coord__('karditeknomo')
            [10, 0, 17, 3, 8, 19, 4, 10, 13, 14, 12, 14]
        """
        return [IFN.node_index(char) for char in str_link]


    @staticmethod
    def premier_ifn(C: list) -> list:
        """
        Return the minimum integer IFN regardless of stochastic matrix.

        Parameters:
            C (list): The capacity matrix.

        Returns:
            list: The premier IFN.

        Example:
            >>> import IdealFlow.Network as net
            >>> C = [[0, 2], [3, 0]]
            >>> n = net.IFN()
            >>> premier_flow = n.premier_ifn(C)
            >>> print(premier_flow)
            [[0, 1], [1, 0]]
        """
        size = np.shape(C)
        mR, mC = size[0], size[1]
        list_cycles = IFN.find_all_cycles_in_matrix(C)
        F = np.zeros((mR, mC))
        for cycle in list_cycles:
            F = IFN.change_flow_in_cycle(F, cycle, +1)
        return F


    @staticmethod
    def abs_diff_capacity_flow(C: list, F: list, w1: float = 1, w2: float = 1) -> float:
        """
        Return scalar cost of the total change between capacity and flow matrix.

        Parameters:
            C (list): The capacity matrix.
            F (list): The flow matrix.
            w1 (float): Weight for negative difference. Default is 1.
            w2 (float): Weight for positive difference. Default is 1.

        Returns:
            float: The total cost.

        Example:
            >>> import IdealFlow.Network as net
            >>> C = [[0, 2], [3, 0]]
            >>> F = [[0, 1], [2, 0]]
            >>> n = net.IFN()
            >>> cost = n.abs_diff_capacity_flow(C, F)
            >>> print(cost)
            2.0
        """
        C = np.array(C)
        F = np.array(F)
        diff = C - F
        cost = np.sum(w2 * diff[diff > 0]) - np.sum(w1 * diff[diff < 0])
        return cost


    @staticmethod
    def is_edge_in_cycle(i: int, j: int, cycle: str) -> bool:
        """
        Check if an edge is in a cycle.

        Parameters:
            i (int): The row index.
            j (int): The column index.
            cycle (str): The cycle string.

        Returns:
            bool: True if the edge is in the cycle, otherwise False.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> cycle = 'abc'
            >>> result = n.is_edge_in_cycle(0, 1, cycle)
            >>> print(result)
            True
        """
        n = len(cycle)
        for k in range(n):
            from_node = ord(cycle[k]) - 97
            to_node = ord(cycle[(k + 1) % n]) - 97
            if from_node == i and to_node == j:
                return True
        return False


    @staticmethod
    def adjacency_to_stochastic(A: list) -> list:
        """
        Convert adjacency matrix to stochastic matrix of equal outflow distribution.

        Parameters:
            A (list): The adjacency matrix.

        Returns:
            list: The stochastic matrix.

        Example:
            >>> import IdealFlow.Network as net
            >>> A = [[0, 2], [3, 0]]
            >>> n = net.IFN()
            >>> stochastic_matrix = n.adjacency_to_stochastic(A)
            >>> print(stochastic_matrix)
            [[0, 1], [1, 0]]
        """
        v=np.sum(A,axis=1)           # node out degree
        D=np.diag(v)                 # degree matrix
        return np.dot(np.linalg.inv(D),A) # ideal flow of equal outflow distribution


    @staticmethod
    def adjacency_to_ideal_flow(A: list, kappa: float = 1) -> list:
        """
        Convert adjacency matrix to ideal flow matrix.

        Parameters:
            A (list): The adjacency matrix.
            kappa (float): The kappa parameter.

        Returns:
            list: The ideal flow matrix.

        Example:
            >>> import IdealFlow.Network as net
            >>> A = [[0, 2], [3, 0]]
            >>> n = net.IFN()
            >>> ideal_flow = n.adjacency_to_ideal_flow(A)
            >>> print(ideal_flow)
        """
        S = IFN.adjacency_to_stochastic(A)
        pi = IFN.markov(S, kappa)
        return IFN.ideal_flow(S, pi)

    
    @staticmethod
    def stochastic_to_pi(S: list, kappa: float = 1) -> list:
        """
        Compute Perron vector (phi) from stochastic matrix.

        Parameters:
            S (list): The stochastic matrix.
            kappa (float): The kappa parameter.

        Returns:
            list: The Perron vector.

        Example:
            >>> import IdealFlow.Network as net
            >>> S = [[0.5, 0.5], [0.5, 0.5]]
            >>> n = net.IFN()
            >>> pi = n.stochastic_to_pi(S)
            >>> print(pi)
        
        Alias:
            :meth:`markov`

        See also:        
            :meth:`stationary_markov_chain`

        """
        return IFN.markov(S, kappa)


    @staticmethod
    def stationary_markov_chain(S: list) -> list:
        """
        Compute the stationary distribution of a Markov chain.

        Parameters:
            S (list): The stochastic matrix.

        Returns:
            list: The stationary distribution.

        Example:
            >>> import IdealFlow.Network as net
            >>> S = [[0.5, 0.5], [0.5, 0.5]]
            >>> n = net.IFN()
            >>> stationary_dist = n.stationary_markov_chain(S)
            >>> print(stationary_dist)
        """
        S = np.array(S)
        n = S.shape[0]
        v = np.ones((n, 1)) / n
        threshold = 1e-7
        for _ in range(10):
            u = np.dot(S.T, v)
            if np.linalg.norm(u - v) < threshold:
                break
            v = u
        return v.__flatten__()


    @staticmethod
    def stochastic_to_ideal_flow(S: list, kappa: float = 1) -> list:
        """
        Convert stochastic matrix to ideal flow matrix.
        If matrix size is les than 25, use exact method else, use approximate method which is fast but inaccurate.

        Parameters:
            S (list): The stochastic matrix.
            kappa (float): The kappa parameter.

        Returns:
            list: The ideal flow matrix.

        Example:
            >>> import IdealFlow.Network as net
            >>> S = [[0.5, 0.5], [0.5, 0.5]]
            >>> n = net.IFN()
            >>> ideal_flow = n.stochastic_to_ideal_flow(S)
            >>> print(ideal_flow)
        """
        S = np.array(S)
        mR, mC = S.shape
        if mR != mC:
            return "Stochastic Matrix must be a square matrix."
        if mR < 25:
            # use exact method. Accurate but slow for large matrix.
            pi = IFN.stochastic_to_pi(S, kappa)
            piJt = np.dot(np.array(pi), np.ones((1, mR)))
            return (piJt * S)
            # pi = IFN.markov(S,kappa)
            # F = IFN.ideal_flow(S, pi)
            return F
        else:
            # use approximation method.Inaccurae but fast
            pi = IFN.stationary_markov_chain(S)
            B = np.dot(np.array(pi)[:, np.newaxis], np.ones((1, mR)))
            return (B * S * kappa)

    
    @staticmethod
    def sum_of_row(M: list) -> list:
        """
        Compute the sum of each row in a matrix.

        Parameters:
            M (list): The input matrix.

        Returns:
            list: The row sums.

        Example:
            >>> import IdealFlow.Network as net
            >>> M = [[1, 2], [3, 4]]
            >>> n = net.IFN()
            >>> row_sums = n.sum_of_row(M)
            >>> print(row_sums)
        """
        return np.sum(M, axis=1)


    @staticmethod
    def sum_of_col(M: list) -> list:
        """
        Compute the sum of each column in a matrix.

        Parameters:
            M (list): The input matrix.

        Returns:
            list: The column sums.

        Example:
            >>> import IdealFlow.Network as net
            >>> M = [[1, 2], [3, 4]]
            >>> n = net.IFN()
            >>> col_sums = n.sum_of_col(M)
            >>> print(col_sums)
        """
        return np.sum(M, axis=0)


    @staticmethod
    def alphabet_list(n: int) -> list:
        """
        Convert an integer to its corresponding ASCII character.
        Useful to generate list of nodes.

        Parameters:
            n (int): The number of characters to generate.

        Returns:
            list: A list of characters from 'a' to the n-th character.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> alphabet = n.alphabet_list(3)
            >>> print(alphabet)
            ['a', 'b', 'c']
        """
        return [chr(i) for i in range(ord('a'), ord('a') + n)]
    
    
    
    @staticmethod
    def is_premier_matrix(F: list) -> bool:
        """
        Check if an ideal flow matrix is premier.

        Parameters:
            F (list): The ideal flow matrix.

        Returns:
            bool: True if the matrix is premier, otherwise False.

        Example:
            >>> import IdealFlow.Network as net
            >>> F = [[1, 1], [1, 1]]
            >>> n = net.IFN()
            >>> result = n.is_premier_matrix(F)
            >>> print(result)
        """
        net_signature = IFN.decompose(F)
        return IFN.is_premier_signature(net_signature)


    @staticmethod
    def __scale_array_to_integer_ratios__(arr: list) -> list:
        """
        Scale array to integer ratios.

        Parameters:
            arr (list): The input array.

        Returns:
            list: The scaled array.

        Example:
            >>> import IdealFlow.Network as net
            >>> arr = [0.5, 0.3]
            >>> n = net.IFN()
            >>> scaled_array = n.__scale_array_to_integer_ratios__(arr)
            >>> print(scaled_array)
        """
        fractions = [IFN.decimal_to_fraction(x) for x in arr]
        common_denominator = np.lcm.reduce([frac[1] for frac in fractions])
        return [(frac[0] * common_denominator) // frac[1] for frac in fractions]
   

    @staticmethod
    def random_ifn(num_nodes: int = 5, total_flow: float = 1) -> list:
        """
        Generate a random ideal flow matrix with a given number of nodes and total flow based on network signature.

        Parameters:
            num_nodes (int): Number of nodes. Default is 5.
            total_flow (float): Total flow. Default is 1.

        Returns:
            list: The ideal flow network.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> random_network = n.random_ifn(3, 10)
            >>> print(random_network)
        """
        if total_flow <= 0:
            total_flow = np.random.randint(10, 100)
        signature = IFN.rand_ifn_signature(num_nodes, total_flow)
        return IFN.compose(signature)


    @staticmethod
    def kappa(F: list) -> float:
        """
        Compute the kappa value of a flow matrix.

        Parameters:
            F (list): The flow matrix.

        Returns:
            float: The kappa value.

        Example:
            >>> import IdealFlow.Network as net
            >>> F = [[1, 2], [3, 4]]
            >>> n = net.IFN()
            >>> kappa_value = n.kappa(F)
            >>> print(kappa_value)
        """
        return np.sum(F)


    @property
    def min_flow(self) -> float:
        """
        Return the internal min flow in the network

        Example:
            >>> import IdealFlow.Network as net
            >>> C = [[1, 2], [3, 4]]
            >>> n = net.IFN()
            >>> n.set_matrix(C,['a','b'])
            >>> print(n.min_flow)
        """
        F, _ = self.get_matrix()
        return IFN.min_flow_matrix(F)        


    @staticmethod
    def min_flow_matrix(F: list) -> float:
        """
        Compute the minimum flow value of a flow matrix.

        Parameters:
            F (list): The flow matrix.

        Returns:
            float: The minimum flow value.

        Example:
            >>> import IdealFlow.Network as net
            >>> F = [[1, 2], [3, 0]]
            >>> n = net.IFN()
            >>> min_flow_value = n.min_flow_matrix(F)
            >>> print(min_flow_value)
        """
        f = np.array(F).__flatten__()
        f = f[f > 0]
        return np.min(f)


    @property
    def max_flow(self) -> float:
        """
        Find the internal maximum flow value in the network.

        Returns:
            float: The maximum flow in the network.

        Example:
            >>> import IdealFlow.Network as net
            >>> C = [[1, 2], [3, 4]]
            >>> n = net.IFN()
            >>> n.set_matrix(C,['a','b'])
            >>> print(n.max_flow)
        """
        lst=[]
        for key, val in self.adjList.items():
            if val!={}:
                lst.append(max(val.values()))
        return max(lst)
    

    @staticmethod
    def max_flow_matrix(F: list) -> float:
        """
        Compute the maximum flow value of a flow matrix.

        Parameters:
            F (list): The flow matrix.

        Returns:
            float: The maximum flow value.

        Example:
            >>> import IdealFlow.Network as net
            >>> F = [[1, 2], [3, 4]]
            >>> n = net.IFN()
            >>> max_flow_value = n.max_flow_matrix(F)
            >>> print(max_flow_value)
        """
        f = np.array(F).__flatten__()
        f = f[f > 0]
        return np.max(f)


    @property
    def average_flow(self) -> float:
        """
        Return the internal average flow in the network

        Example:
            >>> import IdealFlow.Network as net
            >>> C = [[1, 2], [3, 4]]
            >>> n = net.IFN()
            >>> n.set_matrix(C,['a','b'])
            >>> print(n.average_flow)
        """
        F, _ = self.get_matrix()
        return IFN.average_flow_matrix(F) 
    

    @staticmethod
    def average_flow_matrix(F: list) -> float:
        """
        Compute the average flow value of a flow matrix.

        Parameters:
            F (list): The flow matrix.

        Returns:
            float: The average flow value.

        Example:
            >>> import IdealFlow.Network as net
            >>> F = [[1, 2], [3, 4]]
            >>> n = net.IFN()
            >>> avg_flow_value = n.average_flow_matrix(F)
            >>> print(avg_flow_value)
        """
        f = np.array(F).__flatten__()
        f = f[f > 0]
        return np.mean(f)
    
    
    @property
    def stdev_flow(self) -> float:
        """
        Calculate the internal standard deviation of the flow across all links in the network.

        Returns:
            float: The standard deviation of flow.

        Example:
            >>> import IdealFlow.Network as net
            >>> C = [[1, 2], [3, 4]]
            >>> n = net.IFN()
            >>> n.set_matrix(C,['a','b'])
            >>> print(n.stdev_flow)
        """
        m=self.total_links
        avg=self.total_flow/m
        std=0
        count=0
        for startNode in self.adjList.keys(): 
            toNodes=self.out_neighbors(startNode)
            for endNode,weight in toNodes.items():
                std=std+(weight-avg)**2
                count=count+1
        if std>0 and count>0:
            std=math.sqrt(std)/count
        else:
            std=0
        return std    


    @staticmethod
    def stdev_flow_matrix(F: list) -> float:
        """
        Compute the standard deviation of flow values in a flow matrix.

        Parameters:
            F (list): The flow matrix.

        Returns:
            float: The standard deviation of flow values.

        Example:
            >>> import IdealFlow.Network as net
            >>> F = [[1, 2], [3, 4]]
            >>> n = net.IFN()
            >>> std_flow_value = n.stdev_flow_matrix(F)
            >>> print(std_flow_value)
        """
        f = np.array(F).__flatten__()
        f = f[f > 0]
        return np.std(f)


    @property
    def cov_flow(self) -> float:
        """
        Compute the internal coefficient of variation of flows in the network.

        Returns:
            float: The coefficient of variation of flow.

        Example:
            >>> import IdealFlow.Network as net
            >>> C = [[1, 2], [3, 4]]
            >>> n = net.IFN()
            >>> n.set_matrix(C,['a','b'])
            >>> print(n.cov_flow)
        """
        return self.stdev_flow/(self.total_flow/self.total_links)

    
    @staticmethod
    def cov_flow_matrix(F: list) -> float:
        """
        Compute the coefficient of variation of flow values in a flow matrix.

        Parameters:
            F (list): The flow matrix.

        Returns:
            float: The coefficient of variation.

        Example:
            >>> import IdealFlow.Network as net
            >>> F = [[1, 2], [3, 4]]
            >>> n = net.IFN()
            >>> coef_var = n.cov_flow_matrix(F)
            >>> print(coef_var)
        """
        f = np.array(F).flatten()
        f = f[f > 0]
        return np.std(f) / np.mean(f)
    
    

    @staticmethod
    def capacity_to_congestion(C: list, kappa: float, capacity_multiplier: float) -> list:
        """
        Compute congestion matrix from capacity matrix and kappa.

        Parameters:
            C (list): The capacity matrix.
            kappa (float): The kappa parameter.
            capacity_multiplier (float): The capacity multiplier.

        Returns:
            list: The congestion matrix.

        Example:
            >>> import IdealFlow.Network as net
            >>> C = [[3, 2], [5, 0]]
            >>> n = net.IFN()
            >>> congestion = n.capacity_to_congestion(C, 1.5, 0.75)
            >>> print(congestion)
        """
        S = IFN.capacity_to_stochastic(C)
        F = IFN.stochastic_to_ideal_flow(S, kappa)
        return (np.array(F) / (np.array(C) * capacity_multiplier))


    @staticmethod
    def stochastic_to_probability(S: list) -> list:
        """
        Compute probability matrix from stochastic matrix.

        Parameters:
            S (list): The stochastic matrix.

        Returns:
            list: The probability matrix.

        Example:
            >>> import IdealFlow.Network as net
            >>> S = [[0.3, 0.7], [0.6, 0.4]]
            >>> n = net.IFN()
            >>> prob_matrix = n.stochastic_to_probability(S)
            >>> print(prob_matrix)
        """
        phi = IFN.stochastic_to_pi(S, 1)
        phiJt = np.dot(np.array(phi), np.ones((1, len(S))))
        return (S * phiJt)

        
    @staticmethod
    def stochastic_to_network_entropy(S: list) -> float:
        """
        Compute network entropy from stochastic matrix.

        Parameters:
            S (list): The stochastic matrix.

        Returns:
            float: The network entropy.

        Example:
            >>> import IdealFlow.Network as net
            >>> S = [[0.5, 0.5], [0.5, 0.5]]
            >>> n = net.IFN()
            >>> entropy = n.stochastic_to_network_entropy(S)
            >>> print(entropy)
        """
        # s = np.array(S).__flatten__()
        # s = s[s > 0]
        # return -np.sum(s * np.log(s))
        s=S[np.nonzero(S)]
        return np.sum(np.multiply(-s,np.log(s)),axis=None)

        
    @staticmethod
    def stochastic_to_entropy_ratio(S: list) -> float:
        """
        Compute entropy ratio from stochastic matrix.

        Parameters:
            S (list): The stochastic matrix.

        Returns:
            float: The entropy ratio.

        Example:
            >>> import IdealFlow.Network as net
            >>> S = [[0.4, 0.6], [0.7, 0.3]]
            >>> n = net.IFN()
            >>> ratio = n.stochastic_to_entropy_ratio(S)
            >>> print(ratio)
        """
        h1 = IFN.stochastic_to_network_entropy(S)
        A = np.array(S) > 0
        T = IFN.adjacency_to_stochastic(A)
        h0 = IFN.stochastic_to_network_entropy(T)
        if h0>0:
            return h1/h0
        else:
            return 0


    @staticmethod
    def max_network_entropy(P: list) -> tuple:
        """
        Compute maximum network entropy for a given probability matrix.

        Parameters:
            P (list): The probability matrix.

        Returns:
            tuple: The entropy, entropy ratio, and maximum entropy.

        Example:
            >>> import IdealFlow.Network as net
            >>> P = [[0.25, 0.75], [0.6, 0.4]]
            >>> n = net.IFN()
            >>> max_entropy = n.max_network_entropy(P)
            >>> print(max_entropy)
        """
        P = np.array(P)
        arr_prob = P[P > 0]
        n = len(arr_prob)
        p_uniform = 1 / n
        entropy = -np.sum(arr_prob * np.log(arr_prob))
        max_ent = -np.sum([p_uniform * np.log(p_uniform) for _ in range(n)])
        return entropy, entropy / max_ent, max_ent
   
            
    @property
    def network_entropy(self) -> float:
        """
        Calculate the network entropy based on the stochastic flow matrix.

        Returns:
            float: The network entropy.

        Example:
            >>> import IdealFlow.Network as net
            >>> F = [[0, 2], [2, 1]]
            >>> n = net.IFN()
            >>> n.set_matrix(F,['a','b'])
            >>> n.network_entropy
        """
        F,_=self.get_matrix()
        S=self.ideal_flow_to_stochastic(F)
        return self.stochastic_to_network_entropy(S)


    @staticmethod
    def network_entropy_matrix(F: list) -> float:
        """
        Compute network entropy for a given flow matrix.

        Parameters:
            F (list): The flow matrix.

        Returns:
            float: The network entropy.

        Example:
            >>> import IdealFlow.Network as net
            >>> F = [[2, 3], [1, 4]]
            >>> n = net.IFN()
            >>> entropy = n.network_entropy_matrix(F)
            >>> print(entropy)
        """
        total_flow = IFN.kappa(F)
        entropy = 0
        for row in F:
            for val in row:
                if val > 0:
                    p = val / total_flow
                    entropy -= p * np.log(p)
        return entropy


    @property
    def average_node_entropy(self) -> float:
        """
        Compute average node entropy from stochastic matrix.

        Parameters:
            S (list): The stochastic matrix.

        Returns:
            float: The average node entropy.

        Example:
            >>> import IdealFlow.Network as net
            >>> F = [[0, 2], [2, 1]]
            >>> n = net.IFN()
            >>> n.set_matrix(F,['a','b'])
            >>> avg_entropy = n.average_node_entropy
            >>> print(avg_entropy)
        """
        F, _ = self.get_matrix()
        return IFN.average_node_entropy_matrix(F)


    @staticmethod
    def average_node_entropy_matrix(F: list) -> float:
        """
        Compute average node entropy for a given flow matrix.

        Parameters:
            F (list): The flow matrix.

        Returns:
            float: The average node entropy.

        Example:
            >>> import IdealFlow.Network as net
            >>> F = [[2, 3], [1, 4]]
            >>> n = net.IFN()
            >>> avg_entropy = n.average_node_entropy_matrix(F)
            >>> print(avg_entropy)
        """
        C = np.array(F)
        S = IFN.capacity_to_stochastic(C)
        S = np.array(S)
        positive_list = S[S > 0]
        if len(positive_list) == 0:
            return None
        total = np.sum(positive_list)
        probabilities = positive_list / total
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy / S.shape[0]
        

    @property
    def average_node_entropy_ratio(self) -> float:
        """
        Return the internal average node entropy ratio of flows in the network

        Example:
            >>> import IdealFlow.Network as net
            >>> C = [[1, 2], [3, 4]]
            >>> n = net.IFN()
            >>> n.set_matrix(C,['a','b'])
            >>> print(n.average_node_entropy_ratio)
        """
        F, _ = self.get_matrix()
        return IFN.average_node_entropy_ratio_matrix(F) 
    

    @staticmethod
    def average_node_entropy_ratio_matrix(F: list) -> float:
        """
        Compute average node entropy ratio for a given flow matrix.

        Parameters:
            F (list): The flow matrix.

        Returns:
            float: The average node entropy ratio.

        Example:
            >>> import IdealFlow.Network as net
            >>> F = [[2, 3], [1, 4]]
            >>> n = net.IFN()
            >>> ratio = n.average_node_entropy_ratio_matrix(F)
            >>> print(ratio)
        """
        F = np.array(F)
        actual_entropy = IFN.average_node_entropy_matrix(F)
        total_nodes = F.shape[0]
        max_entropy = 0
        for row in F:
            active_connections = np.sum(row > 0)
            if active_connections > 0:
                max_entropy += np.log(active_connections)
        return actual_entropy / (max_entropy / total_nodes)


    @property
    def network_entropy_ratio(self) -> float:
        """
        Return the internal network entropy ratio of flows in the network

        Example:
            >>> import IdealFlow.Network as net
            >>> C = [[1, 2], [3, 4]]
            >>> n = net.IFN()
            >>> n.set_matrix(C,['a','b'])
            >>> print(n.network_entropy_ratio)
        """
        F, _ = self.get_matrix()
        return IFN.network_entropy_ratio_matrix(F) 
    

    @staticmethod
    def network_entropy_ratio_matrix(F: list) -> float:
        """
        Compute network entropy ratio for a given flow matrix.

        Parameters:
            F (list): The flow matrix.

        Returns:
            float: The network entropy ratio.

        Example:
            >>> import IdealFlow.Network as net
            >>> F = [[2, 3], [1, 4]]
            >>> n = net.IFN()
            >>> ratio = n.network_entropy_ratio_matrix(F)
            >>> print(ratio)
        """
        total_flows = np.sum(np.array(F) > 0)
        actual_entropy = IFN.network_entropy_matrix(F)
        max_network_entropy = np.log(total_flows)
        return actual_entropy / max_network_entropy



    '''
    
    Signature Analysis

    '''


    @staticmethod
    def is_equal_signature(signature1: str, signature2: str) -> bool:
        """
        Check if two signatures are equal by comparing their canonical cycle dictionaries.

        Parameters:
            signature1 (str): The first network signature.
            signature2 (str): The second network signature.

        Returns:
            bool: True if the signatures are equal, otherwise False.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> sig1 = "abc + bcd"
            >>> sig2 = "bcd + abc"
            >>> result = n.is_equal_signature(sig1, sig2)
            >>> print(result)  # Output: True
        """
        cycle_dict1 = IFN.parse_terms_to_dict(signature1)
        canon_cycle_dict1 = IFN.canonize_cycle_dict(cycle_dict1)
        cycle_dict2 = IFN.parse_terms_to_dict(signature2)
        canon_cycle_dict2 = IFN.canonize_cycle_dict(cycle_dict2)
        return canon_cycle_dict1 == canon_cycle_dict2

    

    @staticmethod
    def __is_equal_sets__(a: set, b: set) -> bool:
        """
        Check if two sets are equal.

        Parameters:
            a (set): The first set.
            b (set): The second set.

        Returns:
            bool: True if the sets are equal, otherwise False.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> set1 = {1, 2, 3}
            >>> set2 = {3, 2, 1}
            >>> result = n.__is_equal_sets__(set1, set2)
            >>> print(result)  # Output: True
        """
        return a == b


    @staticmethod
    def extract_first_k_terms(net_signature: str, k: int) -> str:
        """
        Extract the first `k` terms from a network signature.

        Parameters:
            net_signature (str): The network signature.
            k (int): Number of terms to extract.

        Returns:
            str: The extracted terms.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> signature = "abc + bcd + cde"
            >>> result = n.extract_first_k_terms(signature, 2)
            >>> print(result)  # Output: "abc + bcd"
        """
        parts = net_signature.split('+')
        return ' + '.join(parts[:k])


    @staticmethod
    def extract_last_k_terms(net_signature: str, k: int) -> str:
        """
        Extract the last `k` terms from a network signature.

        Parameters:
            net_signature (str): The network signature.
            k (int): Number of terms to extract.

        Returns:
            str: The extracted terms.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> signature = "abc + bcd + cde"
            >>> result = n.extract_last_k_terms(signature, 2)
            >>> print(result)  # Output: "bcd + cde"
        """
        parts = net_signature.split('+')
        return ' + '.join(parts[-k:])


    @staticmethod
    def generate_random_terms(net_signature: str, k: int, is_premier: bool = False) -> str:
        """
        Generate `k` random terms from a network signature.

        Parameters:
            net_signature (str): The network signature.
            k (int): Number of terms to generate.
            is_premier (bool): Whether the terms should be premier. Default is False.

        Returns:
            str: The generated terms.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> signature = "abc + bcd + cde"
            >>> result = n.generate_random_terms(signature, 2)
            >>> print(result)
        """
        cycle_dict = IFN.parse_terms_to_dict(net_signature)
        terms = list(cycle_dict.keys())
        np.random.shuffle(terms)
        selected_terms = terms[:k]
        if is_premier:
            return ' + '.join(selected_terms)
        return ' + '.join(f"{np.random.randint(1, 11)}{term}" for term in selected_terms)


    @staticmethod
    def premier_signature(C: list) -> str:
        """
        Compute the premier signature for a given capacity matrix.

        Parameters:
            C (list of list of int/float): The capacity matrix.

        Returns:
            str: The premier signature.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> C = [[3, 1], [1, 0]]
            >>> signature = n.premier_signature(C)
            >>> print(signature)
            ab + a
        """
        cycles = IFN.find_all_cycles_in_matrix(C)
        return ' + '.join(cycles)


    @staticmethod
    def canonize_signature(signature: str) -> str:
        """
        Canonize a network signature by relabeling the nodes and sorting the cycles.

        Parameters:
            signature (str): The network signature.

        Returns:
            str: The canonized signature.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> signature = "bca + cab"
            >>> n.canonize_signature(signature)
            2abc
        """
        # # create node map and relabel in case there is node jump
        # node_mapping = IFN.create_node_mapping(signature)
        # signature2 = IFN.relabel_net_signature(signature, node_mapping)

        # core
        cycle_dict = IFN.parse_terms_to_dict(signature)
        canon_cycle_dict = IFN.canonize_cycle_dict(cycle_dict)
        return IFN.cycle_dict_to_signature(canon_cycle_dict)
        
        # put back to the original node name
        return IFN.reverse_relabel_signature(signature3)
    


    @staticmethod
    def create_node_mapping(net_signature: str) -> dict:
        """
        Create a node mapping for a network signature.
        When the signature contains jump of nodes, we need to create node mapping. 

        Parameters:
            net_signature (str): The network signature.

        Returns:
            dict: The node mapping.

        See also:
            :meth:`relabel_signature`
            :meth:`reverse_relabel_signature`


        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> signature = "abc + bcd"
            >>> mapping = n.create_node_mapping(signature)
            >>> print(mapping)
        """
        unique_nodes = IFN.identify_unique_nodes(net_signature)
        node_mapping = {node: i for i, node in enumerate(unique_nodes)}        
        return node_mapping


    @staticmethod
    def relabel_signature(net_signature: str, node_mapping: dict) -> str:
        """
        Relabel a network signature using a node mapping.

        Parameters:
            net_signature (str): The network signature.
            node_mapping (dict): The node mapping.

        Returns:
            str: The relabeled signature.

        See also:
            :meth:`create_node_mapping`
            :meth:`reverse_relabel_signature`

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> signature = "abc + bcd"
            >>> node_mapping = {'a': 'x', 'b': 'y', 'c': 'z'}
            >>> relabeled = n.relabel_signature(signature, node_mapping)
            >>> print(relabeled)  # Output: "xyz + yzd"
        """
        return ''.join(node_mapping.get(char, char) for char in net_signature)


    @staticmethod
    def reverse_relabel_signature(relabeled_signature: str, node_mapping: dict) -> str:
        """
        Reverse relabel a network signature using a node mapping.

        Parameters:
            relabeled_signature (str): The relabeled signature.
            node_mapping (dict): The node mapping.

        Returns:
            str: The original signature.

        See also:
            :meth:`create_node_mapping`
            :meth:`relabel_signature`

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> relabeled = "xyz + yzd"
            >>> node_mapping = {'x': 'a', 'y': 'b', 'z': 'c'}
            >>> original = n.reverse_relabel_signature(relabeled, node_mapping)
            >>> print(original)  # Output: "abc + bcd"
        """
        inverse_node_mapping = {v: k for k, v in node_mapping.items()}
        return ''.join(inverse_node_mapping.get(char, char) for char in relabeled_signature)


    @staticmethod
    def cycle_dict_to_signature(cycle_dict: dict) -> str:
        """
        Convert a cycle dictionary to a network signature.
        The keys of cycle_dict input must be in canonical cycle.
        The signature is a string of the form "abc + bcd" where a, b, c, d are nodes.

        Parameters:
            cycle_dict (dict): The canonical cycle dictionary.

        Returns:
            str: The network signature.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> cycle_dict = {'abc': 1, 'bcd': -1}
            >>> signature = n.cycle_dict_to_signature(cycle_dict)
            >>> print(signature)             
            'abc + (-bcd)'
            >>> cycle_dict = {'ac': 1, 'acd': 3, 'bc': 2}
            >>> n.cycle_dict_to_signature(cycle_dict)
            'ac + 3acd + 2bc'
        """
        signature = ""
        cycles = list(cycle_dict.keys())
        for index, cycle in enumerate(cycles):
            alpha = cycle_dict[cycle]
            if alpha != 0:
                if index > 0 and index < len(cycles):
                    prevAlpha = cycle_dict[cycles[index - 1]]
                    if prevAlpha == 0:
                        signature += ""
                    else:
                        signature += " + "

                if alpha == 1:
                    signature += f"{cycle}"
                elif alpha == -1:
                    signature += f"(-{cycle})"
                elif alpha < 0:
                    signature += f"({alpha}){cycle}"
                else:
                    signature += f"{alpha}{cycle}"
        return signature        


    @staticmethod
    def is_valid_signature(signature: str) -> bool:
        """
        Check if a network signature is valid by ensuring that all cycles are canonical.

        Parameters:
            signature (str): The network signature.

        Returns:
            bool: True if the signature is valid, otherwise False.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()            
            >>> n.is_valid_signature("abc + bcd")
            True
            >>> n.is_valid_signature("5abc+2bcd")
            True
            >>> n.is_valid_signature("bac + dcb")
            False
            >>> n.is_valid_signature("cbcd")
            False
        """
        cycle_dict = IFN.parse_terms_to_dict(signature)
        canon_cycle_dict = IFN.canonize_cycle_dict(cycle_dict)
        return all(IFN.is_cycle_canonical(cycle) for cycle in canon_cycle_dict)


    @staticmethod
    def is_cycle_canonical(cycle: str) -> bool:
        """
        Check if a cycle is canonical by comparing lexicographically sorted and rotated versions.

        Parameters:
            cycle (str): The cycle string.

        Returns:
            bool: True if the cycle is canonical, otherwise False.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> cycle = "abc"
            >>> n.is_cycle_canonical(cycle)
            True
            >>> cycle = "cab"
            >>> n.is_cycle_canonical(cycle)
            False
            >>> cycle = "bca"
            >>> n.is_cycle_canonical(cycle)
            False

        """
        return all(cycle[i] <= cycle[i + 1] for i in range(len(cycle) - 1))


    @staticmethod
    def is_irreducible_signature(signature: str) -> bool:
        """
        Check if a network signature is irreducible, meaning all cycles share nodes.

        Parameters:
            signature (str): The network signature.

        Returns:
            bool: True if the signature is irreducible, otherwise False.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> signature = "abc + bcd"
            >>> result = n.is_irreducible_signature(signature)
            >>> print(result)  # Output: True
        """
        cycle_dict = IFN.parse_terms_to_dict(signature)
        canon_cycle_dict = IFN.canonize_cycle_dict(cycle_dict)
        cycles = list(canon_cycle_dict.keys())
        if len(cycles) <= 1:
            return True
        return all(
            any(IFN.has_pivot(cycle, other_cycle) for j, other_cycle in enumerate(cycles) if i != j)
            for i, cycle in enumerate(cycles)
        )


    @staticmethod
    def has_pivot(cycle1: str, cycle2: str) -> bool:
        """
        Check if two cycles have a pivot (shared node or link, or path).

        Parameters:
            cycle1 (str): The first cycle.
            cycle2 (str): The second cycle.

        Returns:
            bool: True if there is a pivot, otherwise False.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> cycle1 = "abc"
            >>> cycle2 = "bcd"
            >>> result = n.has_pivot(cycle1, cycle2)
            >>> print(result)  # Output: True
        """
        nodes1 = set(cycle1)
        nodes2 = set(cycle2)
        return bool(nodes1 & nodes2)


    @staticmethod
    def find_pivots(signature: str) -> list:
        """
        Find pivots in a network signature, which are common nodes between cycles.
        
        Parameters:
            signature (str): The network signature.

        Returns:
            list of dict: A list of pivots between cycles.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> signature = "abc + bcd + cde"
            >>> pivots = n.find_pivots(signature)
            >>> print(pivots)
        """
        cycle_dict = IFN.parse_terms_to_dict(signature)
        canon_cycle_dict = IFN.canonize_cycle_dict(cycle_dict)
        cycles = list(canon_cycle_dict.keys())
        pivots = []
        for i, cycle in enumerate(cycles):
            for j, other_cycle in enumerate(cycles):
                if i < j:
                    pivot_type = IFN.find_pivot_type(cycle, other_cycle)
                    if pivot_type != 'no pivot':
                        pivots.append({'pivot': pivot_type, 'cycles': [cycle, other_cycle]})
        return pivots


    @staticmethod
    def find_pivot_type(cycle1: str, cycle2: str) -> str:
        """
        Determine the type of pivot (common node or path) between two cycles.
        
        Parameters:
            cycle1 (str): The first cycle.
            cycle2 (str): The second cycle.

        Returns:
            str: The type of pivot (node, link or path).

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> cycle1 = "abc"
            >>> cycle2 = "bcd"
            >>> result = n.find_pivot_type(cycle1, cycle2)
            >>> print(result)  # Output: 'link: bc'
        """
        common_nodes = set(cycle1) & set(cycle2)
        if len(common_nodes) == 1:
            return 'node: ' + common_nodes.pop()
        elif len(common_nodes) == 2:
            return 'link: ' + ''.join(common_nodes)
        elif len(common_nodes) > 1:
            return 'path: ' + ''.join(common_nodes)
        return 'no pivot'


    @staticmethod
    def canonize_cycle_dict(cycle_dict: dict) -> dict:
        """
        Canonize a cycle dictionary by sorting and normalizing the cycles.
        
        Parameters:
            cycle_dict (dict): The cycle dictionary.

        Returns:
            dict: The canonized cycle dictionary.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> cycle_dict = {'bca': 1, 'cab': 2}
            >>> n.canonize_cycle_dict(cycle_dict)
            {'abc': 3}
        """
        canon_cycle_dict = {}
        for cycle, alpha in cycle_dict.items():
            canon_cycle = IFN.canonize(cycle)
            if canon_cycle in canon_cycle_dict:
                canon_cycle_dict[canon_cycle] += alpha
            else:
                canon_cycle_dict[canon_cycle] = alpha
        return canon_cycle_dict


    @staticmethod
    def canonize(cycle: str) -> str:
        """
        Canonize a string cycle by rotating it to its lexicographically smallest form.
        
        Parameters:
            cycle (str): The string cycle, each letter represent a node.

        Returns:
            str: The canonized string cycle.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> cycle = 'bca'
            >>> canonized = n.canonize(cycle)
            abc
            >>> n.canonize("fde")
            def
        """
        min_idx = min(range(len(cycle)), key=lambda i: cycle[i:] + cycle[:i])
        rotated = cycle[min_idx:] + cycle[:min_idx]
        reversed_cycle = rotated[::-1]
        return min(rotated, reversed_cycle)


    @staticmethod
    def link_cycle_matrix(F: list) -> dict:
        """
        Generate the link-cycle matrix from a flow matrix.
        
        Parameters:
            F (list of list of int/float): The flow matrix.

        Returns:
            dict: A dictionary containing the link-cycle matrix, cycles, and links.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> F = [[3, 1], [1, 0]]
            >>> result = n.link_cycle_matrix(F)
            >>> print(result)
        """
        n = len(F)
        A = np.array(F) != 0
        cycles = IFN.find_all_cycles_in_matrix(A)
        links = [(i, j) for i in range(n) for j in range(n) if F[i][j] != 0]
        H = np.zeros((len(links), len(cycles)))
        for link_idx, (i, j) in enumerate(links):
            for cycle_idx, cycle in enumerate(cycles):
                cycle_nodes = [ord(c) - 97 for c in cycle]
                if i in cycle_nodes and j in cycle_nodes:
                    pos_i = cycle_nodes.index(i)
                    if cycle_nodes[(pos_i + 1) % len(cycle_nodes)] == j:
                        H[link_idx, cycle_idx] = 1
        y = [F[i][j] for i, j in links]
        return {'H': H, 'y': y, 'cycles': cycles, 'links': links}


    @staticmethod
    def find_all_cycles_in_matrix(matrix: list) -> list:
        """
        Return list of all possible cycles from the weighted adjacency matrix as cycle terms.
        
        Parameters:
            matrix (list of list of int/float): The input matrix.

        Returns:
            list of tuple of str: A list of cycles terms found in the matrix.

        See also:
            :meth:`find_all_cycles_in_adj_list`
        
        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> matrix = [[0, 3, 0],[2, 0, 1],[1, 0, 0]]
            >>> n.find_all_cycles_in_matrix(matrix)
            ['ab', 'abc']            
        """
        n = len(matrix)
        adjList = [[] for _ in range(n)]
        for i, row in enumerate(matrix):
            for j, cell in enumerate(row):
                if cell != 0:
                    adjList[i].append(j)

        cycles = set()

        def canonical(cycle):
            rotations = [cycle[i:] + cycle[:i] for i in range(len(cycle))]
            min_rotation = min(rotations)
            reversed_rotation = min_rotation[::-1]
            return ''.join(min_rotation) if ''.join(min_rotation) < ''.join(reversed_rotation) else ''.join(reversed_rotation)

        def dfs(v, start, visited, stack):
            visited[v] = True
            stack.append(v)
            for w in adjList[v]:
                if w == start and len(stack) >= 1:
                    cycles.add(canonical([IFN.node_name(node) for node in stack]))
                elif not visited[w]:
                    dfs(w, start, visited, stack)
            stack.pop()
            visited[v] = False

        for i in range(n):
            visited = [False] * n
            dfs(i, i, visited, [])

        return list(cycles)    


    @staticmethod
    def find_all_cycles_in_adj_list(adjL: dict) -> list:
        """
        Find all cycles in a given adjacency list.
        
        Parameters:
            adjL (dict): The adjacency list.

        Returns:
            list of str: The list of cycles found in the adjacency list.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> adjL = {'a': {'b': 1}, 'b': {'c': 1}, 'c': {'a': 1}}
            >>> cycles = n.find_all_cycles_in_adj_list(adjL)
            >>> print(cycles)
            [('a', 'b', 'c')] 
            >>> adj_list={0: {1: 3}, 1: {0: 2, 2: 1}, 2: {0: 1}}
            >>> n.find_all_cycles_in_adj_list(adj_list)
            [(0, 1), (0, 1, 2)]
            >>> adjL={0: {0: 3, 1: 1}, 1: {0: 1}}
            >>> n.find_all_cycles_in_adj_list(adjL)
            [(0,), (0, 1)]
        """
        nodes = list(adjL.keys())
        node_index = {node: i for i, node in enumerate(nodes)}
        adj_list = [[] for _ in range(len(nodes))]
        for from_node, targets in adjL.items():
            for to_node, value in targets.items():
                if value is not None:
                    adj_list[node_index[from_node]].append(node_index[to_node])
        cycles = set()
        def canonical(cycle):
            min_idx = min(range(len(cycle)), key=lambda i: cycle[i:] + cycle[:i])
            rotated = cycle[min_idx:] + cycle[:min_idx]
            reversed_cycle = rotated[::-1]
            return min(rotated, reversed_cycle)
        def dfs(v, start, visited, stack):
            visited[v] = True
            stack.append(v)
            for w in adj_list[v]:
                if w == start and len(stack) >= 1:
                    # cycles.add(tuple(canonical([nodes[node] for node in stack])))
                    cycles.add(canonical([nodes[node] for node in stack]))
                elif not visited[w]:
                    dfs(w, start, visited, stack)
            stack.pop()
            visited[v] = False
        
        # Run DFS from every node
        for i in range(len(nodes)):
            if adj_list[i]:
                # dfs(i, i, [False] * len(nodes), [])
                IFN.dfs_adj_list(i, i, [False] * len(nodes), [], adj_list, cycles, nodes)

        return list(cycles)


    @staticmethod
    def find_all_walks_in_matrix(matrix: list) -> list:
        """
        Find all walks in a given matrix.
        
        Parameters:
            matrix (list of list of int/float): The input matrix.

        Returns:
            list of str: A list of walks found in the matrix.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> matrix = [[1, 1], [1, 0]]
            >>> walks = n.find_all_walks_in_matrix(matrix)
            >>> print(walks)  # Output: ['ab']
        """
        n = len(matrix)
        nodes = [chr(97 + i) for i in range(n)]
        walks = []
        def find_walk(start_row):
            walk = ''
            current_row = start_row
            while True:
                found = False
                for col in range(n):
                    if matrix[current_row][col] != 0:
                        walk += nodes[current_row]
                        matrix[current_row][col] = 0
                        current_row = col
                        found = True
                        break
                if not found:
                    walk += nodes[current_row]
                    break
            return walk if len(walk) > 1 else None
        for row in range(n):
            while True:
                walk = find_walk(row)
                if walk:
                    walks.append(walk)
                else:
                    break
        return walks


    @staticmethod
    def compose(signature: str) -> list:
        """
        Compose a flow matrix from a network signature.
        
        Parameters:
            signature (str): The network signature.

        Returns:
            list of list of int/float: The composed flow matrix.

        See also:
            :meth:`string_to_matrix`

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> signature = "abc"
            >>> n.compose(signature)
            array([[0., 1., 0.],
                [0., 0., 1.],
                [1., 0., 0.]])
            >>> n.compose('ab+ba+cab')
            array([[0., 3., 0.],
                [2., 0., 1.],
                [1., 0., 0.]])
        """
        terms = IFN.parse_terms_to_dict(signature)
        num_nodes = IFN.signature_to_num_nodes(signature)
        F = np.zeros((num_nodes, num_nodes),dtype=int)
        for cycle, coef in terms.items():
            cycle_nodes = IFN.__string_link_to_coord__(cycle)
            IFN.assign_cycle_to_matrix(F, cycle_nodes, coef)
        return F


    @staticmethod
    def decompose(matrix: list) -> str:
        """
        Decompose a ideal flow matrix into a network signature.
        
        Parameters:
            F (list of list of int/float): The flow matrix.

        Returns:
            str: The decomposed network signature.
        
        Raises:
            ValueError: If F is not an ideal flow matrix.

        See also:
            :meth:`compose`
            :meth:`find_all_cycles_in_matrix`

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> F = [[3, 1], [1, 0]]
            >>> signature = n.decompose(F)
            >>> print(signature)
            3a + ab
            >>> F=[[0, 3, 0], [2, 0, 1], [1, 0, 0]]
            >>> n.decompose(F)
            abc + 2ab

        """
        if not IFN.is_ideal_flow_matrix(matrix):
            raise ValueError("Method decompose only can accept an ideal flow matrix as input")
        
        F = copy.deepcopy(matrix) # to remove by reference side effect

        cycles = IFN.find_all_cycles_in_matrix(F) 
        # print('cycles:',cycles)
        cycle_dict = {}
        for cycle in cycles:
            cycle_value = min(IFN.flows_in_cycle(F, cycle))
            if cycle_value > 0:
                cycle_dict[cycle] = cycle_value
                F = IFN.change_flow_in_cycle(F, cycle, -cycle_value)
        return IFN.cycle_dict_to_signature(cycle_dict)


    @staticmethod
    def flow_matrix_to_adj_list(F: list) -> dict:
        """
        Convert a flow matrix to an adjacency list.
        
        Parameters:
            F (list of list of int/float): The flow matrix.

        Returns:
            dict: The adjacency list representation of the flow matrix.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> F = [[3, 1], [1, 0]]
            >>> adj_list = n.flow_matrix_to_adj_list(F)
            >>> print(adj_list)
            {0: {0: 3, 1: 1}, 1: {0: 1}}
            >>> F=[[0, 3, 0], [2, 0, 1], [1, 0, 0]]
            >>> n.flow_matrix_to_adj_list(F)
            {0: {1: 3}, 1: {0: 2, 2: 1}, 2: {0: 1}}
        """
        adj_list = {}
        for i, row in enumerate(F):
            for j, value in enumerate(row):
                if value != 0:
                    if i not in adj_list:
                        adj_list[i] = {}
                    adj_list[i][j] = value
        return adj_list


    @staticmethod
    def _find_common_nodes(cycle1: str, cycle2: str) -> list:
        """
        Find common nodes between two cycles.

        Parameters:
            cycle1 (str): The first cycle.
            cycle2 (str): The second cycle.

        Returns:
            list of str: A list of common nodes.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> cycle1 = "abc"
            >>> cycle2 = "bcd"
            >>> result = n._find_common_nodes(cycle1, cycle2)
            >>> print(result)  # Output: ['b', 'c']
        """
        return list(set(cycle1) & set(cycle2))


    @staticmethod
    def is_premier_signature(net_signature: str) -> bool:
        """
        Check if a network signature is premier (has all cycle and coefficients equal to 1).

        Parameters:
            net_signature (str): The network signature.

        Returns:
            bool: True if the signature is premier, otherwise False.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> signature = "abc + def"
            >>> n.is_premier_signature(signature)
            True
            >>> n.is_premier_signature("abc + 2ab")
            False
            >>> n.is_premier_signature("abc + bca + def")
            False
        """
        signature1 = IFN.signature_coef_to_1(net_signature)
        F = IFN.compose(signature1)
        cycles = IFN.find_all_cycles_in_matrix(F)
        cycle_dict = IFN.parse_terms_to_dict(net_signature)
        canon_cycle_dict = IFN.canonize_cycle_dict(cycle_dict)
        cycle_set = set(cycles)
        return all(cycle_set and canon_cycle_dict[cycle] == 1 for cycle in canon_cycle_dict) and len(cycle_set) == len(canon_cycle_dict)


    @staticmethod
    def signature_coef_to_1(net_signature: str) -> str:
        """
        Convert all network signature coefficients to one.
        Internally, it also canonize the cycle terms.

        Parameters:
            net_signature (str): The network signature.

        Returns:
            str: The converted signature with all coefficients set to one.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> signature = "2bca + 3fde"
            >>> result = n.signature_coef_to_1(signature)
            >>> print(result)  # Output: "abc + def"
        """
        cycle_dict = IFN.parse_terms_to_dict(net_signature)
        canon_cycle_dict = IFN.canonize_cycle_dict(cycle_dict)
        return ' + '.join(canon_cycle_dict)


    @staticmethod
    def assign_cycle_to_matrix(F: np.ndarray, cycle: list, value: int) -> None:
        """
        Assign a value to the edges in a cycle within the flow matrix.
        return F by reference

        Parameters:
            F (np.ndarray): The flow matrix.
            cycle (list): The cycle as a list of node indices.
            value (number): The value to assign to the edges.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> F = np.zeros((3, 3), dtype=int)
            >>> n.assign_cycle_to_matrix(F, [0, 1, 2], 5)
            >>> F
            array([[0, 5, 0],
                   [0, 0, 5],
                   [5, 0, 0]])
            >>> F = [[0, 0], [0, 0]]
            >>> cycle = [0, 1]
            >>> n.assign_cycle_to_matrix(F, cycle, 1)
            [[0, 1], [1, 0]]
        """
        for i in range(len(cycle) - 1):
            F[cycle[i]][cycle[i + 1]] += value
        F[cycle[-1]][cycle[0]] += value


    @staticmethod
    def identify_unique_nodes(signature):
        """
        Identify the unique nodes in a network signature.

        Parameters:
            net_signature (str): The network signature.

        Returns:
            list of str: A list of unique nodes.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> signature = "abc + def"
            >>> result = n.identify_unique_nodes(signature)
            >>> print(result)  # Output: ['a', 'b', 'c', 'd', 'e', 'f']
        """
        unique_nodes = set()
        parts = signature.split('+')
        for part in parts:
            cycle_str = ''.join(filter(str.isalpha, part))
            unique_nodes.update(cycle_str)
        return sorted(unique_nodes)    
    

    @staticmethod
    def signature_to_link_flow(cycle_signature: str, is_cycle: bool = True) -> dict:
        """
        Compute link flow values from a cycle signature.

        Parameters:
            cycle_signature (str): The cycle signature.
            is_cycle (bool): Whether the signature represents a cycle.

        Returns:
            dict: The link flow values.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> signature = "abc + def"
            >>> result = n.signature_to_link_flow(signature)
            >>> print(result)  # Output: {'ab': 1, 'bc': 1, 'ca': 1, 'de': 1, 'ef': 1, 'fd': 1}
        """
        terms = IFN.parse_terms_to_dict(cycle_signature)
        link_flows = {}
        for cycle, coef in terms.items():
            nodes = cycle
            for i in range(len(nodes)):
                current = nodes[i]
                next_ = nodes[(i + 1) % len(nodes)] if is_cycle else nodes[i + 1] if i + 1 < len(nodes) else None
                if next_:
                    link = current + next_
                    link_flows[link] = link_flows.get(link, 0) + coef
        return link_flows


    @staticmethod
    def signature_to_num_nodes(cycle_signature: str) -> int:
        """
        Compute the number of nodes in a cycle signature.

        Parameters:
            cycle_signature (str): The cycle signature.

        Returns:
            int: The number of unique nodes.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> signature = "abc + def"
            >>> result = n.signature_to_num_nodes(signature)
            >>> print(result)  # Output: 6
        """
        return len(set(char for term in cycle_signature.split('+') for char in term if char.isalpha()))


    @staticmethod
    def signature_to_links(cycle_signature: str, is_cycle: bool = True) -> set:
        """
        Compute the links in a cycle signature.

        Parameters:
            cycle_signature (str): The cycle signature.
            is_cycle (bool): Whether the signature represents a cycle.

        Returns:
            set: A set of links.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> signature = "abc + def"
            >>> result = n.signature_to_links(signature)
            >>> print(result)  # Output: {'ab', 'bc', 'ca', 'de', 'ef', 'fd'}
        """
        terms = IFN.parse_terms_to_dict(cycle_signature)
        unique_links = set()
        for cycle in terms:
            nodes = cycle
            for i in range(len(nodes)):
                link = nodes[i] + nodes[(i + 1) % len(nodes)] if is_cycle else nodes[i] + nodes[i + 1] if i + 1 < len(nodes) else None
                if link:
                    unique_links.add(link)
        return unique_links


    @staticmethod
    def signature_to_num_links(cycle_signature: str, is_cycle: bool = True) -> int:
        """
        Compute the number of links in a cycle signature.

        Parameters:
            cycle_signature (str): The cycle signature.
            is_cycle (bool): Whether the signature represents a cycle.

        Returns:
            int: The number of links.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()        
            >>> n.signature_to_num_links("abc + def")
            4
        """
        return len(IFN.signature_to_links(cycle_signature, is_cycle))


    @staticmethod
    def signature_to_row_stochastic(cycle_signature: str, is_cycle: bool = True) -> dict:
        """
        Convert cycle signature to a row stochastic adjacency list.

        Parameters:
            cycle_signature (str): The cycle signature.
            is_cycle (bool): Whether the signature represents a cycle.

        Returns:
            dict: The row stochastic adjacency list.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()        
            >>> n.signature_to_row_stochastic("abc")
            {'a': {'b': 1.0}, 'b': {'c': 1.0}, 'c': {'a': 1.0}}
        """
        link_flows = IFN.signature_tolink_flow(cycle_signature, is_cycle)
        row_sums = IFN.signature_to_sum_rows(cycle_signature)
        stocastic_adj_list = {}
        for link, flow in link_flows.items():
            source, target = link
            if source not in stocastic_adj_list:
                stocastic_adj_list[source] = {}
            stocastic_adj_list[source][target] = str(flow / row_sums[source])
        return stocastic_adj_list


    @staticmethod
    def signature_to_column_stochastic(cycle_signature: str, is_cycle: bool = True) -> dict:
        """
        Convert cycle signature to a column stochastic adjacency list.

        Parameters:
            cycle_signature (str): The cycle signature.
            is_cycle (bool): Whether the signature represents a cycle.

        Returns:
            dict: The column stochastic adjacency list.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()        
            >>> n.signature_to_column_stochastic("abc")
            {'a': {'b': 1.0}, 'b': {'c': 1.0}, 'c': {'a': 1.0}}
        """
        link_flows = IFN.signature_tolink_flow(cycle_signature, is_cycle)
        col_sums = IFN.signature_to_sum_cols(cycle_signature)
        stocastic_adj_list = {}
        for link, flow in link_flows.items():
            source, target = link
            if source not in stocastic_adj_list:
                stocastic_adj_list[source] = {}
            stocastic_adj_list[source][target] = str(flow / col_sums[target])
        return stocastic_adj_list


    @staticmethod
    def signature_to_ideal_flow(cycle_signature: str, is_cycle: bool = True) -> dict:
        """
        Convert cycle signature to an ideal flow adjacency list.

        Parameters:
            cycle_signature (str): The cycle signature.
            is_cycle (bool): Whether the signature represents a cycle.

        Returns:
            dict: The ideal flow adjacency list.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()        
            >>> n.signature_to_ideal_flow("abc")
            {'a': {'b': 1.0}, 'b': {'c': 1.0}, 'c': {'a': 1.0}}
        """
        return IFN.signature_to_adj_list(cycle_signature, is_cycle)


    @staticmethod
    def signature_to_kappa(cycle_signature: str) -> float:
        """
        Compute the kappa value of a cycle signature.

        Parameters:
            cycle_signature (str): The cycle signature.

        Returns:
            float: The kappa value.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()        
            >>> n.signature_to_kappa("abc + def")
            
        """
        terms = IFN.parse_terms_to_dict(cycle_signature)
        return sum(alpha * len(cycle) for cycle, alpha in terms.items())


    @staticmethod
    def signature_to_coef_flow(cycle_signature: str) -> float:
        """
        Compute the coefficient of flow of a cycle signature.

        Parameters:
            cycle_signature (str): The cycle signature.

        Returns:
            float: The coefficient of flow.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()        
            >>> n.signature_to_coef_flow("abc + def")            
        """
        link_flows = IFN.signature_to_link_flow(cycle_signature)
        kappa = IFN.signature_to_kappa(cycle_signature)
        avg_flow = kappa / len(link_flows)
        variance = sum((flow - avg_flow) ** 2 for flow in link_flows.values()) / len(link_flows)
        return (variance ** 0.5) / avg_flow


    @staticmethod
    def signature_to_max_flow(cycle_signature: str) -> float:
        """
        Compute the maximum flow value in a cycle signature.

        Parameters:
            cycle_signature (str): The cycle signature.

        Returns:
            float: The maximum flow value.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()        
            >>> n.signature_to_max_flow("abc + def")
            1.0
        """
        adjL = IFN.signature_to_ideal_flow(cycle_signature)
        return max(link_flow for node in adjL for link_flow in adjL[node].values())


    @staticmethod
    def signature_to_min_flow(cycle_signature: str) -> float:
        """
        Compute the minimum flow value in a cycle signature.

        Parameters:
            cycle_signature (str): The cycle signature.

        Returns:
            float: The minimum flow value.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()        
            >>> n.signature_to_min_flow("abc + def")
            1.0
        """
        adjL = IFN.signature_to_ideal_flow(cycle_signature)
        return min(link_flow for node in adjL for link_flow in adjL[node].values())


    @staticmethod
    def signature_to_sum_rows(cycle_signature: str) -> dict:
        """
        Compute the sum of rows in a cycle signature.

        Parameters:
            cycle_signature (str): The cycle signature.

        Returns:
            dict: A dictionary of row sums.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()        
            >>> n.signature_to_sum_rows("abc + def")
            {'a': 1, 'b': 1, 'c': 1, 'd': 1, 'e': 1, 'f': 1}
        """
        terms = IFN.parse_terms_to_dict(cycle_signature)
        row_sums = {}
        for cycle, coef in terms.items():
            for node in cycle:
                row_sums[node] = row_sums.get(node, 0) + coef
        return row_sums


    @staticmethod
    def signature_to_sum_cols(cycle_signature: str) -> dict:
        """
        Compute the sum of columns in a cycle signature.

        Parameters:
            cycle_signature (str): The cycle signature.

        Returns:
            dict: A dictionary of column sums.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()        
            >>> n.signature_to_sum_cols("abc + def")
            {'a': 1, 'b': 1, 'c': 1, 'd': 1, 'e': 1, 'f': 1}
        """
        return IFN.signature_to_sum_rows(cycle_signature)


    @staticmethod
    def signature_to_pivots(cycle_signature: str) -> dict:
        """
        Find pivots in a cycle signature.

        Parameters:
            cycle_signature (str): The cycle signature.

        Returns:
            dict: A dictionary of pivots between cycles.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()        
            >>> n.signature_to_pivots("abc + abd")
            {'abc-abd': ['a', 'b']}
        """
        terms = IFN.parse_terms_to_dict(cycle_signature)
        cycles = list(terms.keys())
        pivots = {}
        for i, cycle in enumerate(cycles):
            for j in range(i + 1, len(cycles)):
                common_nodes = IFN._find_common_nodes(cycle, cycles[j])
                if common_nodes:
                    pivots[f"{cycle}-{cycles[j]}"] = common_nodes
        return pivots


    @staticmethod
    def is_irreducible_signature(cycle_signature: str) -> bool:
        """
        Check if a cycle signature is irreducible.

        Parameters:
            cycle_signature (str): The cycle signature.

        Returns:
            bool: True if the signature is irreducible, False otherwise.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()        
            >>> n.is_irreducible_signature("abc + abd")
            False
        """
        terms = IFN.parse_terms_to_dict(cycle_signature)
        cycles = list(terms.keys())
        for i, cycle in enumerate(cycles):
            if not any(IFN._find_common_nodes(cycle, cycles[j]) for j in range(len(cycles)) if i != j):
                return False
        return True


    # @staticmethod
    # def rand_ifn_signature(num_nodes: int = 4, kappa: int = 17) -> str:
    #     """
    #     Generate a random IFN signature.

    #     Parameters:
    #         num_nodes (int): The number of nodes.
    #         kappa (int): The kappa value.

    #     Returns:
    #         str: The generated IFN signature.

    #     Example:
    #         >>> import IdealFlow.Network as net
    #         >>> n = net.IFN()        
    #         >>> n.rand_ifn_signature(4, 17)
    #         'abcd'
    #     """
    #     nodes = [chr(97 + i) for i in range(num_nodes)]
    #     first_cycle = ''.join(nodes)
    #     first_cycle_length = len(first_cycle)
    #     cycle_dict = {first_cycle: 1}
    #     remaining_kappa = kappa - first_cycle_length
    #     count_loop = 0
    #     while remaining_kappa > 0:
    #         existing_cycles = list(cycle_dict.keys())
    #         random_cycle = existing_cycles[np.random.randint(len(existing_cycles))]
    #         pivot_start = np.random.randint(len(random_cycle))
    #         pivot_length = np.random.randint(1, len(random_cycle) - pivot_start + 1)
    #         pivot = random_cycle[pivot_start:pivot_start + pivot_length]
    #         new_cycle_nodes = set(pivot)
    #         while len(new_cycle_nodes) < pivot_length + np.random.randint(0, num_nodes - pivot_length + 1):
    #             new_cycle_nodes.add(nodes[np.random.randint(num_nodes)])
    #         if len(new_cycle_nodes) == 1 and count_loop < 100:
    #             count_loop += 1
    #             continue
    #         new_cycle = IFN.canonize(new_cycle_nodes)
    #         new_cycle_length = len(new_cycle)
    #         new_cycle_coefficient = remaining_kappa // new_cycle_length
    #         if new_cycle_coefficient > 0:
    #             cycle_dict[new_cycle] = cycle_dict.get(new_cycle, 0) + new_cycle_coefficient
    #             remaining_kappa -= new_cycle_coefficient * new_cycle_length
    #         elif new_cycle_length <= remaining_kappa:
    #             cycle_dict[new_cycle] = cycle_dict.get(new_cycle, 0) + 1
    #             remaining_kappa -= new_cycle_length
    #     return IFN.cycle_dict_to_signature(cycle_dict)
    
    def rand_ifn_signature(self, numNodes=5, kappa=17):
        """
        Generate a random cycle signature for an Ideal Flow Network (IFN) that meets the specified number of nodes and total flow.

        This method constructs a cycle signature by randomly generating cycles and assigning coefficients to them,
        ensuring that the total flow (sum of coefficients times cycle lengths) equals the specified kappa.

        Parameters:
            numNodes (int, optional): The number of nodes in the network. Default is 5.
            kappa (int, optional): The total flow in the network. Default is 100.

        Returns:
            str: The cycle signature as a string.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> signature = n.findIFNsignature(numNodes=5, kappa=50)
            >>> print(signature)
            abcde + 6abce + 4bcd + 2ade

            The output is a cycle signature string where each term represents a cycle,
            and the coefficient indicates the number of times the cycle occurs in the network.

        Notes:
            - The method ensures that the total flow does not exceed the specified kappa.
            - It avoids adding self-loops (cycles of length 1) unless necessary.
        """
        # Generate node names
        nodes = [self.node_name(i) for i in range(numNodes)]
        firstCycle = ''.join(nodes)
        firstCycleLength = len(firstCycle)
        cycleDict = {}
        remainingKappa = kappa - firstCycleLength  # The first cycle has coefficient 1
        countLoop = 0

        # Add the first cycle to the cycle dictionary with coefficient 1
        cycleDict[firstCycle] = 1

        # Continue adding cycles until the remaining kappa is exhausted
        while remainingKappa > 0:
            # Select a random pivot from the existing cycles
            existingCycles = list(cycleDict.keys())
            randomCycle = random.choice(existingCycles)
            pivotStart = random.randint(0, len(randomCycle) - 1)
            pivotLength = random.randint(1, len(randomCycle) - pivotStart)
            pivot = randomCycle[pivotStart:pivotStart + pivotLength]

            # Create a new cycle by adding random nodes around the pivot
            newCycleNodes = set(pivot)
            maxAdditionalNodes = numNodes - len(newCycleNodes)
            additionalNodes = random.randint(0, maxAdditionalNodes)
            while len(newCycleNodes) < len(pivot) + additionalNodes:
                randomNode = random.choice(nodes)
                newCycleNodes.add(randomNode)

            if len(newCycleNodes) == 1 and countLoop < 10:
                countLoop += 1
                continue  # Skip this iteration to avoid self-loop

            newCycleArray = list(newCycleNodes)
            newCycle = ''.join(self.canonize(newCycleArray))
            newCycleLength = len(newCycle)

            # Determine the coefficient for the new cycle
            newCycleCoefficient = remainingKappa // newCycleLength

            if newCycleCoefficient > 0:
                cycleDict[newCycle] = cycleDict.get(newCycle, 0) + newCycleCoefficient
                remainingKappa -= newCycleCoefficient * newCycleLength
            else:
                # If newCycleCoefficient is 0, add the cycle with coefficient 1 if it can still fit in the remainingKappa
                if newCycleLength <= remainingKappa:
                    cycleDict[newCycle] = cycleDict.get(newCycle, 0) + 1
                    remainingKappa -= newCycleLength
                else:
                    # Cannot fit any more cycles
                    break

        return self.cycle_dict_to_signature(cycleDict)
    

    @staticmethod
    def cardinal_ifn_signature(A: list) -> str:
        """
        Find the cardinal IFN signature, which represents the minimal set of cycles that can fully describe the network.

        Parameters:
            A (list of list of float): The input matrix representing the flow between nodes.

        Returns:
            str: The cardinal IFN signature.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> A = [[0, 1, 0], [0, 0, 1], [1, 0, 0]]
            >>> n.cardinal_ifn_signature(A)
            'abc'
        """
        all_cycles = IFN.find_all_cycles_in_matrix(A)
        sorted_cycles = sorted(all_cycles, key=len, reverse=True)
        lookup_set = IFN.signature_to_links(' + '.join(all_cycles))
        selected_cycles = []
        current_links = set()
        for cycle in sorted_cycles:
            cycle_links = IFN.signature_to_links(cycle)
            new_links = cycle_links - current_links
            if new_links:
                selected_cycles.append(cycle)
                current_links |= new_links
        selected_signature = ' + '.join(selected_cycles)
        return selected_signature if current_links == lookup_set else None


    @staticmethod
    def find_cardinal_ifn_signature_exhaustive(A: list) -> str:
        """
        Find the cardinal IFN signature using exhaustive search.

        Parameters:
            A (list of list of float): The input matrix representing the flow between nodes.

        Returns:
            str: The cardinal IFN signature.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> A = [[0, 1, 0], [0, 0, 1], [1, 0, 0]]
            >>> n.find_cardinal_ifn_signature_exhaustive(A)
            'abc'
        """
        all_cycles = IFN.find_all_cycles_in_matrix(A)
        all_combinations = IFN.generate_combinations(all_cycles)
        lookup_set = IFN.signature_to_links(' + '.join(all_cycles))
        min_flow = float('inf')
        cardinal_signature = None
        for combination in all_combinations:
            signature = ' + '.join(combination)
            links = IFN.signature_to_links(signature)
            if links == lookup_set:
                flow = IFN.signature_to_kappa(signature)
                if flow < min_flow:
                    min_flow = flow
                    cardinal_signature = signature
        return cardinal_signature


    @staticmethod
    def assign_adjacency_list(value: float, trajectory: str, is_cycle: bool = True) -> dict:
        """
        Assign a value to a sequence of nodes, creating an adjacency list.

        Parameters:
            value (float): The value to assign to the edges in the sequence.
            trajectory (str): The sequence of nodes.
            is_cycle (bool): Whether the sequence represents a cycle (default is True).

        Returns:
            dict: The adjacency list representation of the trajectory.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> n.assign_adjacency_list(1.0, "abc")
            {'a': {'b': 1.0}, 'b': {'c': 1.0}, 'c': {'a': 1.0}}
        """        
        adj_list = {}
        for i in range(len(trajectory)):
            current = trajectory[i]
            next_ = trajectory[(i + 1) % len(trajectory)] if is_cycle else trajectory[i + 1] if i + 1 < len(trajectory) else None
            if next_:
                if current not in adj_list:
                    adj_list[current] = {}
                adj_list[current][next_] = adj_list[current].get(next_, 0) + value
        return adj_list


    @staticmethod
    def merge_adjacency_list(adj_list1: dict, adj_list2: dict) -> dict:
        """
        Merge two adjacency lists by combining their weights.

        Parameters:
            adj_list1 (dict): The first adjacency list.
            adj_list2 (dict): The second adjacency list.

        Returns:
            dict: The merged adjacency list.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> adj_list1 = {'a': {'b': 1}}
            >>> adj_list2 = {'b': {'c': 2}}
            >>> n.merge_adjacency_list(adj_list1, adj_list2)
            {'a': {'b': 1}, 'b': {'c': 2}}
        """
        merged_adj_list = adj_list1.copy()
        for node, targets in adj_list2.items():
            if node not in merged_adj_list:
                merged_adj_list[node] = {}
            for target, value in targets.items():
                merged_adj_list[node][target] = merged_adj_list[node].get(target, 0) + value
        return merged_adj_list


    @staticmethod
    def merge_signatures(sig1: str, sig2: str) -> str:
        """
        Merge two network signatures into one.

        Parameters:
            sig1 (str): The first network signature.
            sig2 (str): The second network signature.

        Returns:
            str: The merged network signature.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> n.merge_signatures('abc', 'def')
            'abc + def'
        """
        if not sig1:
            return sig2
        if not sig2:
            return sig1
        return f"{sig1} + {sig2}"


    @staticmethod
    def signature_to_adj_list(signature: str, is_cycle: bool = True) -> dict:
        """
        Convert a network signature into an adjacency list.

        Parameters:
            signature (str): The network signature.
            is_cycle (bool): Whether the signature represents a cycle (default is True).

        Returns:
            dict: The adjacency list representation of the signature.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> n.signature_to_adj_list("abc")
            {'a': {'b': 1.0}, 'b': {'c': 1.0}, 'c': {'a': 1.0}}
        """
        adj_list = {}
        terms = IFN.parse_terms_to_dict(signature)
        for cycle, coef in terms.items():
            partial_adj_list = IFN.assign_adjacency_list(coef, cycle, is_cycle)
            adj_list = IFN.merge_adjacency_list(adj_list, partial_adj_list)
        return adj_list



    '''
       PRIVATE METHODS
    '''
    
    def __updateNetworkProbability__(self) -> None:
        """
        Update the network probability based on the total flow of the network.

        The method multiplies the weights of each link in the adjacency list by a factor that corresponds
        to the inverse of the total flow, normalizing the probabilities.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> n.total_flow = 5
            >>> n.adjList = {'a': {'b': 1}, 'b': {'c': 1}}
            >>> n.__updateNetworkProbability__()
            >>> n.network_prob
            {'a': {'b': 0.2}, 'b': {'c': 0.2}}
        """
        kappa=self.total_flow 
        if kappa>0:
            adjList=IFN.copy_dict(self.adjList)
            updatedAdjList=self.__updateAdjList__(adjList, 1 / kappa)
            self.network_prob=updatedAdjList
        

    def __updateAdjList__(self, adjList: dict, factor: float) -> dict:
        """
        Multiply the weights in an adjacency list by a scalar factor.
        This method acts like a matrix scalar multiplication, updating the adjacency list.

        Parameters:
            adjList (dict): The adjacency list representing the network.
            factor (float): The scalar factor to multiply the weights by.

        Returns:
            dict: The updated adjacency list with weights multiplied by the factor.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> adjList = {'a': {'b': 1.0}, 'b': {'c': 2.0}}
            >>> n.__updateAdjList__(adj_list, 0.5)
            {'a': {'b': 0.5}, 'b': {'c': 1.0}}
        """
        updatedAdjList={}
        for startNode in adjList.keys(): 
            toNodes=adjList[startNode]
            for endNode,weight in toNodes.items():
                toNodes[endNode]=weight*factor
            updatedAdjList[startNode]=toNodes
        return updatedAdjList
    
    
    def __adjList2Matrix__(self, adjList: dict) -> tuple:
        """
        Convert an adjacency list into an adjacency matrix.
        Generic conversion any adjacency list to adjacency matrix, listNode
        Parameters:
            adjList (dict): The adjacency list.

        Returns:
            tuple: A tuple containing the adjacency matrix and the list of nodes.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> adj_list = {'a': {'b': 1}, 'b': {'c': 1}}
            >>> matrix, list_node = n.__adjList2Matrix__(adj_list)
            >>> matrix
            array([[0., 1.], [0., 0.]])
        """
        listNode=self.__adjList2listNode__(adjList)
        n=len(listNode)
        matrix=IFN.__null_matrix__(n,n)
        for row,startNode in enumerate(listNode):
            toNodes=self.out_neighbors(startNode)
            for endNode,weight in toNodes.items():
                col=listNode.index(endNode)
                matrix[row][col]=weight
        return matrix, listNode
    
    
    def __matrix2AdjList__(self, matrix: list, listNode: list) -> dict:
        """
        Convert an adjacency matrix into an adjacency list.

        Parameters:
            matrix (list of list of float): The adjacency matrix representing node connections and weights.
            listNode (list of str): List of node names corresponding to the matrix rows and columns.

        Returns:
            dict: The adjacency list representation of the matrix.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> matrix = [[0, 1, 0], [0, 0, 1], [1, 0, 0]]
            >>> listNode = ['a', 'b', 'c']
            >>> n.__matrix2AdjList__(matrix, listNode)
            {'a': {'b': 1}, 'b': {'c': 1}, 'c': {'a': 1}}
        """
        adjList={}
        for row,rows in enumerate(matrix):
            toNodes={}
            startNode=listNode[row]
            for col, weight in enumerate(rows):
                if weight>0:
                    endNode=listNode[col]
                    toNodes[endNode]=weight
                adjList[startNode]=toNodes
        return adjList
    
    
    def __adjList2listNode__(self, adjList: dict) -> list:
        """
        Generate a list of unique nodes (listNode) from an adjacency list.

        Parameters:
            adjList (dict): The adjacency list.

        Returns:
            list of str: The sorted list of unique nodes.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> adj_list = {'a': {'b': 1}, 'b': {'c': 2}}
            >>> n.__adjList2listNode__(adj_list)
            ['a', 'b', 'c']
        """
        listNode=set() 
        for startNode in adjList.keys():
            listNode.add(startNode)
            toNodes=adjList[startNode]
            for endNode in toNodes.keys():
                listNode.add(endNode)
        listNode=list(listNode)
        listNode.sort()
        return listNode
    
    
    def __getLinkWeight__(self, startNode: str, endNode: str, adjList: dict) -> float:
        """
        Retrieve the weight of a link between two nodes (from startNode to endNode) in an adjacency list.

        Parameters:
            startNode (str): The starting node.
            endNode (str): The ending node.
            adjList (dict): The adjacency list from which to retrieve the weight.

        Returns:
            float: The weight of the link between startNode and endNode, or 0 if no link exists.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> adj_list = {'a': {'b': 1.0}, 'b': {'c': 2.0}}
            >>> n.__getLinkWeight__('a', 'b', adj_list)
            1.0
            >>> n.__getLinkWeight__('b', 'c', adj_list)
            2.0
        """
        try:
            return adjList[startNode][endNode]
        except KeyError:
            return 0  # Link does not exist
    
    
    def __getWeightLink__(self, startNode: str, endNode: str) -> float:
        """
        Get the weight of a link from startNode to endNode in the internal adjacency list.

        Parameters:
            startNode (str): The start node.
            endNode (str): The end node.

        Returns:
            float: The weight of the link. If the link does not exist, returns 0.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> n.adjList = {'a': {'b': 1.0}, 'b': {'c': 2.0}}
            >>> n.__getWeightLink__('a', 'b')
            1.0
        """
        weight=0
        toNodes=self.out_neighbors(startNode)
        if endNode in toNodes:
            weight=toNodes[endNode]
        return  weight
    
    
    def __setWeightLink__(self, startNode: str, endNode: str, weight: float) -> None:
        """
        Set the weight of a link directly in the internal adjacency list. If the link does not exist, it is created.

        Parameters:
            startNode (str): The start node.
            endNode (str): The end node.
            weight (float): The weight of the link to set.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> n.adjList = {'a': {'b': 1.0}}
            >>> n.__setWeightLink__('b', 'c', 2.0)
            >>> n.adjList
            {'a': {'b': 1.0}, 'b': {'c': 2.0}}
        """
        # add startNode and endNode if not exist
        if startNode not in self.listNodes:
            self.add_node(startNode)
        if endNode not in self.listNodes:
            self.add_node(endNode)
            
        if startNode in self.adjList.keys():
            # if startNode exists in adjList
            toNodes=self.adjList[startNode]
            toNodes[endNode]=weight
        else: # if startNode is not yet in adjList
            # create this endNode with weight
            toNodes={endNode: weight}
        self.adjList[startNode]=toNodes
    
    

    @staticmethod
    def sum_dict_values(dic: dict) -> float:
        """
        Sum the values of a dictionary, treating `None` as zero.

        Parameters:
            dic (dict): The dictionary with values to sum.

        Returns:
            float: The sum of the dictionary's values.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()        
            >>> d = {'a': 1, 'b': None, 'c': 3}
            >>> n.sum_dict_values(d)
            4.0
        """
        return sum(v if v is not None else 0 for v in dic.values())
    
        
    @staticmethod
    def __inf_matrix__(mR: int, mC: int) -> list:
        """
        Generate a matrix of size (mR, mC) filled with `math.inf`.

         Parameters:
            mR (int): Number of rows.
            mC (int): Number of columns.

        Returns:
            list: Matrix of size (mR, mC) with `math.inf` as each element.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()        
            >>> n.__inf_matrix__(2, 2)
            [[inf, inf], [inf, inf]]
        """
        return [[math.inf for _ in range(mC)] for _ in range(mR)]
    
    
    @staticmethod
    def __null_matrix__(mR: int, mC: int) -> list:
        """
        Generate a zero matrix of size (mR, mC).

        Parameters:
            mR (int): Number of rows.
            mC (int): Number of columns.

        Returns:
            list: Matrix of size (mR, mC) with zeros.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()        
            >>> n.__null_matrix__(2, 2)
            [[0, 0], [0, 0]]
        """
        return [[0 for _ in range(mC)] for _ in range(mR)]
    
    
    @staticmethod
    def matrix_replace_value(matrix: list, old_value: float, new_value: float) -> list:
        """
        Replace all occurrences of `old_value` with `new_value` in a matrix.

        Parameters:
            matrix (list): The matrix to modify.
            old_value (float): The value to replace.
            new_value (float): The value to use as a replacement.

        Returns:
            list: The modified matrix.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()        
            >>> matrix = [[1, 2], [2, 3]]
            >>> n.matrix_replace_value(matrix, 2, 0)
            [[1, 0], [0, 3]]
        """
        # return [[new_value if x == old_value else x for x in row] for row in matrix]
        # Create a null matrix to avoid modifying the original by reference
        new_matrix = [[0 for _ in range(len(matrix[0]))] for _ in range(len(matrix))]
        
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                new_matrix[i][j] = new_value if matrix[i][j] == old_value else matrix[i][j]

        return new_matrix
    

    @staticmethod
    def inverse_dict(dic: dict) -> dict:
        """
        Create an inverse dictionary where values become keys and keys become values.

        Parameters:
            dic (dict): The original dictionary.

        Returns:
            dict: Inverse dictionary with values as keys and lists of original keys as values.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()        
            >>> dic = {'a': 1, 'b': 2, 'c': 2}
            >>> n.inverse_dict(dic)
            {1: ['a'], 2: ['b', 'c']}

            >>> dic = {'a': 3, 'c': 2, 'b': 2, 'e': 3, 'd': 1, 'f': 2}
            >>> n.inverse_dict(dic)
            {1: ['d'], 2: ['c', 'b', 'f'], 3: ['a', 'e']}
        """
        inv_dict = {}
        for k, v in dic.items():
            inv_dict.setdefault(v, []).append(k)
        return inv_dict
    
    
    @staticmethod
    def copy_dict(dic: dict) -> dict:
        """
        Create a shallow copy of a nested dictionary.

        Parameters:
            dic (dict): The dictionary to copy.

        Returns:
            dict: A copy of the dictionary.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()        
            >>> d = {'a': [1, 2], 'b': {'c': 3}}
            >>> copy_d = n.copy_dict(d)
            >>> copy_d == d
            True
            >>> copy_d is d
            False
        """
        return eval(repr(dic))  # A quick way to clone nested structures but use with caution.
    
       
        
    @staticmethod
    def node_name(index: int) -> str:
        """
        Convert an index back into a node name based on the original node_index logic.
        Generate a node name based on an index, using letters and base62 encoding.

        Parameters:
            index (int): The index to convert to a node name.

        Returns:
            str: The node name.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()        
            >>> n.node_name(0)
            'a'
            >>> n.node_name(26)
            'A'
            >>> n.node_name(52)
            '0'  # Example base62 string for index 52
            >>> n.node_name(682)
            'aa'
        """
        if 0 <= index < 26:
            return chr(index + 97)  # a-z -> 0-25
        elif 26 <= index < 52:
            return chr(index - 26 + 65)  # A-Z -> 26-51
        else:
            # Handle Base62 digits and other characters starting at index 52
            return IFN.to_base62(index - 52)  # base62 string for values 52+


    @staticmethod
    def node_index(name: str) -> int:
        """
        Convert a name to a unique index.
        - Single lowercase letter: returns an index between 0-25.
        - Single uppercase letter: returns an index between 26-51.
        - Single digit or Base62 character: returns its corresponding index starting from 52.
        - Multi-character name: returns a Base62-encoded index after 62.

        Parameters:
            name (str): The name to convert to an index.

        Returns:
            int: The corresponding node index.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()        
            >>> n.node_index('a')
            0
            >>> n.node_index('A')
            26
            >>> n.node_index('0')
            52
            >>> n.node_index('aa')
            682 
        """
        if len(name) == 1:
            if name.islower():
                return ord(name) - 97  # 'a' to 'z' -> 0-25
            elif name.isupper():
                return ord(name) - 65 + 26  # 'A' to 'Z' -> 26-51
            else:
                # Handle digits and other Base62 characters ('0'-'9', 'a'-'z', 'A'-'Z')
                return IFN.from_base62(name) + 52
        else:
            # Multi-character name (Base62 encoding for longer names)
            return IFN.from_base62(name) + 52  # Base62 encoding for multi-character names

        
    @staticmethod
    def from_base62(s: str) -> int:
        """
        Convert a base62 string to a integer number.

        Parameters:
            s (str): The base62 string.

        Returns:
            int: The corresponding number.
        
        Raises:
            ValueError: If the string contains invalid base62 characters.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()        
            >>> n.from_base62('0')
            0
            >>> n.from_base62('A')
            36
            >>> n.from_base62('10')
            62
        """
        chars = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        base = 62
        num = 0
        for char in s:
            if char not in chars:
                raise ValueError(f"Invalid character in base62 string: {char}")
            num = num * base + chars.index(char)
        return num        


    @staticmethod
    def to_base62(num: int) -> str:
        """
        Convert a integer number to a base62 string.

        Parameters:
            num (int): The number to convert.

        Returns:
            str: The base62 string.

        Raises:
            ValueError: If the number is negative.
        
        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()        
            >>> n.to_base62(0)
            '0'
            >>> n.to_base62(36)
            'A'
            >>> n.to_base62(123)
            '1Z'
        """
        if num < 0:
            raise ValueError("Number must be non-negative.")
        chars = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        base = 62
        result = []
        while num > 0:
            result.append(chars[num % base])
            num //= base
        return ''.join(reversed(result)) or '0'


    @staticmethod    
    def num_to_excel_col(num: int) -> str:
        """
        Convert a number to an Excel-style column label.
        such as `a, b, ..., z, aa, ab, ..., az, ba, ...`
        Use it to rename the variable.

        Parameters:
            num (int): The number to convert.

        Returns:
            str: The Excel column label.
        
        Raises:
            ValueError: If the number is less than 1.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()        
            >>> n.num_to_excel_col(1)
            'a'
            >>> n.num_to_excel_col(27)
            'aa'
        """
        if num < 1:
            raise ValueError("Number must be greater than 0.")
        result = ""
        while num > 0:
            num, r = divmod(num - 1, 26)
            result = chr(r + ord('a')) + result
        return result        
    
    
    @staticmethod
    def excel_col_to_num(col: str) -> int:
        """
        Convert an Excel-style column label to a number.

        Parameters:
            col (str): The column label (e.g., 'a', 'aa').

        Returns:
            int: The corresponding number.
        
        Raises:
            ValueError: If the column contains non-alphabetic characters.
        
        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()        
            >>> n.excel_col_to_num('a')
            1
            >>> n.excel_col_to_num('aa')
            27
        """
        if not col.isalpha():
            raise ValueError("Column label must only contain alphabetic characters.")
    
        num = 0
        for c in col:
            num = num * 26 + (ord(c.upper()) - ord('A') + 1)
        return num


    @staticmethod
    def find_element_in_list(element, list_element: list) -> list:
        """
        Find all indices of an element in a list.

        Parameters:
            element: The element to find.
            list_element (list): The list to search in.

        Returns:
            list: List of indices where the element is found.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()        
            >>> n.find_element_in_list(1, [1, 2, 1, 3])
            [0, 2]
        """
        return [i for i, x in enumerate(list_element) if x == element]
    
    
    @staticmethod
    def find_key_in_dict(val, dic: dict) -> str:
        """
        Find the first key in a dictionary that matches a given value.

        Parameters:
            val: The value to search for.
            dic (dict): The dictionary to search in.

        Returns:
            str: The key corresponding to the given value, or `None` if not found.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()        
            >>> d = {'a': 1, 'b': 2}
            >>> n.find_key_in_dict(2, d)
            'b'
            >>> n.find_key_in_dict(2, {'a': 1, 'e': 2, 'c': 2})
            'e'
        """
        return next((k for k, v in dic.items() if v == val), None)

    
    def color_graph(self, M: list, color: list, pos: int, c: int) -> bool:
        """
        Determines if the adjacency matrix can be colored into two colors (bipartite).

        Parameters:
            M (list of list): The adjacency matrix representing the graph.
            color (list): The color array, with -1 representing uncolored nodes.
            pos (int): The current node position.
            c (int): The color to assign (0 or 1).

        Returns:
            bool: True if the graph can be colored with two colors, False otherwise.

        Example:            
            >>> import IdealFlow.Network as net     # import package.module as alias
            >>> n = net.IFN()
            >>> M = [[0, 1], [1, 0]]
            >>> n.set_matrix(M,['a','b'])
            >>> color = [-1] * len(M)
            >>> n.color_graph(M, color, 0, 0)
            True
        """
        V = self.total_nodes
        if color[pos] != -1 and color[pos] != c:
            return False

        color[pos] = c
        for i in range(V):
            if M[pos][i]:
                if color[i] == -1:
                    if not self.color_graph(M, color, i, 1 - c):
                        return False
                elif color[i] == c:
                    return False
        return True 
    
    
    @staticmethod
    def decimal_to_fraction(decimal: float, tolerance: float = 1.0E-6) -> tuple:
        """
        Converts a decimal to its closest fraction using a form of the Stern-Brocot tree approach.

        Parameters:
            decimal (float): The decimal number to convert.
            tolerance (float): Precision tolerance. (default: use 6 decimal digits to make it precise)

        Returns:
            tuple: The numerator and denominator of the fraction.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()        
            >>> n.decimal_to_fraction(0.75)
            (3, 4)
            >>> n.decimal_to_fraction(0.333333)
            (1, 3)
            >>> n.decimal_to_fraction(0.111111)
            (1, 9)
            >>> import numpy as np
            >>> n.decimal_to_fraction(np.pi)
            (355, 113)
        """
        numerator = 1
        denominator = 1
        lower_numerator = 0
        lower_denominator = 1
        upper_numerator = 1
        upper_denominator = 0

        while True:
            middle_numerator = lower_numerator + upper_numerator
            middle_denominator = lower_denominator + upper_denominator

            if middle_denominator * (decimal + tolerance) < middle_numerator:
                upper_numerator = middle_numerator
                upper_denominator = middle_denominator
            elif middle_numerator < (decimal - tolerance) * middle_denominator:
                lower_numerator = middle_numerator
                lower_denominator = middle_denominator
            else:
                numerator = middle_numerator
                denominator = middle_denominator
                break
        return numerator, denominator


    @staticmethod
    def num_to_str_fraction(num: float) -> str:
        """
        Converts a number to a string fraction representation.

        Parameters:
            num (float): The number to convert.

        Returns:
            str: The string representation of the fraction.

        Example:
            >>> import IdealFlow.Network as net
            >>> import numpy as np
            >>> n = net.IFN()        
            >>> n.num_to_str_fraction(0.75)
            '3/4'
            >>> n.num_to_str_fraction(0.111111)
            '1/9'
            >>> n.num_to_str_fraction(np.pi)
            '355/113'
        """
        if num == 0:
            return '0'
        n, d = IFN.decimal_to_fraction(num)
        return f"{n}/{d}" if d != 1 else str(n)


    @staticmethod
    def combinations(N: int) -> list:
        """
        Generates all combinations of the letters 'a' through 'z' up to length N.

        Parameters:
            N (int): The length of the desired combinations.

        Returns:
            list of str: The list of combinations.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()        
            >>> n.combinations(2)
            ['a', 'b', 'ab']
            >>> n.combinations(3)
            ['a', 'b', 'c', 'ab', 'ac', 'bc', 'abc']
        """
        from itertools import combinations as it_combinations
        results = []
        letters = 'abcdefghijklmnopqrstuvwxyz'[:N]

        for r in range(1, len(letters) + 1):
            for combo in it_combinations(letters, r):
                results.append(''.join(combo))

        return results


    @staticmethod
    def permutations(N: int) -> list:
        """
        Generates all non-empty permutations of N letters.

        Parameters:
            N (int): The number of letters to permute.

        Returns:
            list of str: The list of permutations.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()        
            >>> n.permutations(2)
            ['a', 'b', 'ab', 'ba']
            >>> n.permutations(3)
            ['a', 'b', 'c', 'ab', 'ac', 'ba', 'bc', 'ca', 'cb', 'abc', 'acb', 'bac', 'bca', 'cab', 'cba']
        """
        from itertools import permutations as it_permutations

        results = []
        letters = 'abcdefghijklmnopqrstuvwxyz'[:N]

        for r in range(1, N + 1):
            for perm in it_permutations(letters, r):
                results.append(''.join(perm))

        return results


    @staticmethod
    def generate_combinations(elements: list) -> list:
        """
        Generates all combinations of a given list of elements.

        Parameters:
            elements (list): The list of elements.

        Returns:
            list of list: The list of combinations.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()        
            >>> n.generate_combinations(['a', 'b'])
            [['a'], ['a', 'b'], ['b']]
            >>> n.generate_combinations(['a', 'b', 'c'])
            [['a'], ['a', 'b'], ['a', 'b', 'c'], ['a', 'c'], ['b'], ['b', 'c'], ['c']]
        """
        result = []

        def f(prefix, elements):
            for i in range(len(elements)):
                result.append(prefix + [elements[i]])
                f(prefix + [elements[i]], elements[i + 1:])

        f([], elements)
        return result

    

    @staticmethod
    def is_cycle_has_coef_1(cycle_str: str) -> bool:
        """
        Checks if each term in a cycle string has a coefficient of 1.
        It does not combine the term.

        Parameters:
            cycle_str (str): The cycle string, where terms are separated by '+'.

        Returns:
            bool: True if all terms have a coefficient of 1, otherwise False.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()        
            >>> n.is_cycle_has_coef_1('2a+b+c')
            False
            >>> n.is_cycle_has_coef_1('a+b+c')
            True
            >>> n.is_cycle_has_coef_1('a+b+c+a') # not combine the terms
            True
        """
        parts = cycle_str.split('+')
        for part in parts:
            count_str = ''.join(filter(str.isdigit, part.strip()))
            # count_str = ''.join(filter(str.isdigit, part))
            count = int(count_str) if count_str else 1
            
            if count != 1:
                return False
        return True
    

    @staticmethod
    def form_link_cycle_matrix(F: np.ndarray) -> tuple:
        """
        Forms the link-cycle matrix H and the link flow vector y.

        Parameters:
            F (np.ndarray): Flow matrix representing the network.

        Returns:
            tuple: A tuple containing the link-cycle matrix H, the link flow vector y, 
                   a list of cycles, and a list of links.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()        
            >>> F = np.array([[0, 1], [1, 0]])
            >>> H, y, cycles, links = n.form_link_cycle_matrix(F)
        """
        n = len(F)
        A = (F != 0).astype(int)
        cycles = IFN.find_cycles(A)
        links = [(i, j) for i in range(n) for j in range(n) if F[i][j] != 0]
        H = np.zeros((len(links), len(cycles)), dtype=int)

        for link_idx, (i, j) in enumerate(links):
            for cycle_idx, cycle in enumerate(cycles):
                cycle_nodes = [IFN.node_index(c) for c in cycle]
                if i in cycle_nodes and j in cycle_nodes:
                    pos_i = cycle_nodes.index(i)
                    if cycle_nodes[(pos_i + 1) % len(cycle_nodes)] == j:
                        H[link_idx][cycle_idx] = 1

        y = F[F != 0]
        return H, y, cycles, links


    @staticmethod
    def parse_cycle(cycle: str) -> list:
        """
        Parses a cycle string into a list of node indices.

        Parameters:
            cycle (str): The cycle string.

        Returns:
            list: A list of node indices representing the cycle.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()        
            >>> n.parse_cycle('abc')
            [0, 1, 2]
        """
        return [IFN.node_index(name) for name in cycle]


    @staticmethod
    def identify_unique_nodes(signature: str) -> list:
        """
        Identifies unique nodes in a cycle string.

        Parameters:
            signature (str): The cycle string with node names.

        Returns:
            list: A sorted list of unique node names.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()        
            >>> n.identify_unique_nodes('a + b + c + a')
            ['a', 'b', 'c']
        """
        unique_nodes = set()
        parts = signature.split('+')
        for part in parts:
            cycle_str = ''.join(filter(str.isalpha, part))
            unique_nodes.update(cycle_str)
        return sorted(unique_nodes)




    @staticmethod
    def string_to_matrix(signature: str) -> np.ndarray:
        """
        Converts a cycle string into a flow matrix.

        Parameters:
            signature (str): The cycle string representation.

        Returns:
            np.ndarray: The flow matrix.
        
        See also:
            :meth:`compose`

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()        
            >>> F = n.string_to_matrix('a+b+c')
            >>> F
            array([[1, 0, 0],
                   [0, 1, 0],
                   [0, 0, 1]])
            >>> F = n.string_to_matrix('ab+ba+cab')
            array([[0, 3, 0],
                   [2, 0, 1],
                   [1, 0, 0]])       
        """
        unique_nodes = IFN.identify_unique_nodes(signature)
        node_mapping = {node: i for i, node in enumerate(unique_nodes)}
        n = len(unique_nodes)
        F = np.zeros((n, n), dtype=int)
        parts = signature.split('+')

        for part in parts:
            count_str = ''.join(filter(str.isdigit, part.strip()))
            count = int(count_str) if count_str else 1
            cycle_str = ''.join(filter(str.isalpha, part.strip()))
            cycle = [node_mapping[node] for node in cycle_str]
            IFN.assign_cycle_to_matrix(F, cycle, count)

        return F


    @staticmethod
    def solve_cycles(F: np.ndarray, method: str = 'lsq_linear') -> str:
        """
        Solves the cycle decomposition for a given flow matrix using one of three methods:
        'pinv', 'lsq_linear', or 'nnls'.

        Parameters:
            F (np.ndarray): The flow matrix.
            method (str, optional): The method to solve the system. Options are:
                - 'pinv': Uses the generalized inverse solution (can produce negative results).
                - 'lsq_linear': Uses least-squares with non-negativity constraints (default).
                - 'nnls': Uses non-negative least squares.

        Returns:
            str: A string representation of the decomposed cycles.

        Raises:
            ValueError: If an unsupported method is provided.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()        
            >>> F = np.array([[0, 1], [1, 0]])
            >>> n.solve_cycles(F, method='lsq_linear')
            'ab + ba'
        """
        from scipy.linalg import pinv
        from scipy.optimize import lsq_linear, nnls
        # Form the link-cycle matrix H and link flow vector y
        H, y, cycles, links = IFN.form_link_cycle_matrix(F)

        # Solve using the specified method
        if method == 'pinv':
            # Method 1: Solve Hx = y using the generalized inverse (pinv)
            x = np.dot(pinv(H), y)
        elif method == 'lsq_linear':
            # Method 2: Solve Hx = y using least squares with non-negativity constraint
            result = lsq_linear(H, y, bounds=(0, np.inf))
            x = result.x
        elif method == 'nnls':
            # Method 3: Solve Hx = y using non-negative least squares
            x, _ = nnls(H, y)
        else:
            raise ValueError(f"Unsupported method '{method}'. Use 'pinv', 'lsq_linear', or 'nnls'.")

        # Normalize x to have integer values if necessary
        min_nonzero = np.min(np.abs(x[np.nonzero(x)])) if np.any(x) else 1
        if not np.isclose(min_nonzero, round(min_nonzero)):
            x = np.round(x / min_nonzero).astype(int)
        else:
            x = np.round(x).astype(int)

        # Form the string representation of cycle contributions
        cycle_contributions = [
            f"{x[i]}{cycle}" if x[i] != 1 else cycle for i, cycle in enumerate(cycles) if x[i] > 0
        ]
        return " + ".join(cycle_contributions)


   
    
    '''
        
            CYCLE-STRING 
        
    '''
    @staticmethod
    def extract_first_k_terms(cycle_str: str, k: int) -> str:
        """
        Extracts the first k terms from a cycle string.

        Parameters:
            cycle_str (str): The cycle string with terms separated by ' + '.
            k (int): The number of terms to extract.

        Returns:
            str: A new cycle string with the first k terms.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()        
            >>> n.extract_first_k_terms('a + b + c + d', 2)
            'a + b'
        """
        parts = cycle_str.split(' + ')
        return ' + '.join(parts[:k])


    @staticmethod
    def extract_last_k_terms(cycle_str: str, k: int) -> str:
        """
        Extracts the last k terms from a cycle string.

        Parameters:
            cycle_str (str): The cycle string with terms separated by ' + '.
            k (int): The number of terms to extract.

        Returns:
            str: A new cycle string with the last k terms.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()        
            >>> n.extract_last_k_terms('a + b + c + d', 2)
            'c + d'
        """
        parts = cycle_str.split(' + ')
        return ' + '.join(parts[-k:])


    @staticmethod
    def parse_terms_to_dict(signature: str) -> dict:
        """
        Parses a signature into a dictionary with terms as keys and their coefficients as values.

        Parameters:
           signature (str): The cycle string where each term is a variable or has a coefficient.

        Returns:
            dict: A dictionary where keys are terms (variables) and values are their coefficients.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()
            >>> n.parse_terms_to_dict('2a + b + 3c')
            {'a': 2, 'b': 1, 'c': 3}
            >>> n.parse_terms_to_dict("abc + 2bca + 5def")
            {'abc': 1, 'bca': 2, 'def': 5}
        """
        terms = signature.split('+')
        term_dict = {}
        for term in terms:
            term = term.strip()
            count_str = ''.join(filter(str.isdigit, term))
            signature = ''.join(filter(str.isalpha, term))
            if signature not in term_dict:
                term_dict[signature] = int(count_str) if count_str else 1
        return term_dict


    @staticmethod
    def generate_random_terms(cycle_dict: dict, k: int, is_premier: bool = False) -> str:
        """
        Generates a random cycle string with k terms selected from the given cycle dictionary.
        Optionally, assigns random coefficients to the terms.

        Parameters:
            cycle_dict (dict): A dictionary of cycle terms and their coefficients.
            k (int): The number of terms to generate.
            is_premier (bool, optional): If True, returns terms without random coefficients.

        Returns:
            str: A new cycle string with k randomly selected terms.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()        
            >>> cycle_dict = {'a': 2, 'b': 1, 'c': 3}
            >>> n.generate_random_terms(cycle_dict, 2)
            '2a + 1b'
        """
        terms = list(cycle_dict.keys())
        random.shuffle(terms)
        selected_terms = terms[:k]
        result_terms = []
        for term in selected_terms:
            if is_premier:
                result_terms.append(term)
            else:
                random_coefficient = random.randint(1, 10)  # Adjust range as needed
                if random_coefficient == 1:
                    result_terms.append(term)
                else:
                    result_terms.append(f"{random_coefficient}{term}")
        return ' + '.join(result_terms)

     
    @staticmethod
    def rand_int(mR: int, mC: int, max_val: int = 10, prob: float = 0.8) -> np.ndarray:
        """
        Generate a random integer matrix with biased zero entries.

        Parameters:
            mR (int): Number of rows.
            mC (int): Number of columns.
            max_val (int): Maximum value for non-zero entries. Defaults to 10.
            prob (float): Probability of zeros. Defaults to 0.8.

        Returns:
            np.ndarray: A random integer matrix.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()        
            >>> m = n.rand_int(3, 3, max_val=5, prob=0.7)
            >>> print(m)
            [[0 3 0]
             [4 0 2]
             [0 0 5]]
        """
        if not (0 <= prob <= 1):
            raise ValueError("Probability 'prob' must be between 0 and 1.")
        m = np.zeros((mR, mC), dtype=int)
        for r in range(mR):
            for c in range(mC):
                random_num = random.randint(1, max_val)
                if random_num > max_val / 2 and random.random() < prob:
                    random_num = 0
                m[r][c] = random_num
        return m


    @staticmethod
    def rand_stochastic(n: int) -> np.ndarray:
        """
        Generate a random row-stochastic matrix of size n.

        Parameters:
            n (int): Size of the square matrix.

        Returns:
            np.ndarray: The random row-stochastic matrix.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()        
            >>> S = n.rand_stochastic(3)
            >>> print(S)
            [[0.2 0.3 0.5]
             [0.4 0.2 0.4]
             [0.3 0.3 0.4]]
        """
        C = IFN.rand_int(n, n)
        mJ = np.ones((n, n))
        mDenom = C @ mJ
        with np.errstate(divide='ignore', invalid='ignore'):
            S = np.divide(C, mDenom, where=(mDenom != 0))
        return S


    @staticmethod
    def is_row_stochastic_matrix(mA: np.ndarray) -> bool:
        """
        Check if a matrix is row-stochastic (rows sum to 1).

        Parameters:
            mA (np.ndarray): The input matrix.

        Returns:
            bool: True if the matrix is row-stochastic, otherwise False.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()        
            >>> mA = np.array([[0.5, 0.5], [0.3, 0.7]])
            >>> n.is_row_stochastic_matrix(mA)
            True
        """
        mA = np.array(mA)
        if mA.shape[0] != mA.shape[1]:
            return False

        row_sums = mA.sum(axis=1)
        return np.allclose(row_sums, 1)


    @staticmethod
    def adj_list_to_matrix(adjL: dict) -> np.ndarray:
        """
        Convert an adjacency list to a weighted square matrix.

        Parameters:
            adjL (dict): The adjacency list.

        Returns:
            np.ndarray: The weighted square matrix.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()        
            >>> adjL = {'a': {'b': 1, 'c': 2}, 'b': {'c': 3}, 'c': {}}
            >>> matrix = n.adj_list_to_matrix(adjL)
            >>> print(matrix)
            [[0 1 2]
             [0 0 3]
             [0 0 0]]
        """
        nodes = sorted(set(adjL.keys()).union(*(adjL[node].keys() for node in adjL)))
        size = len(nodes)
        node_index = {node: idx for idx, node in enumerate(nodes)}
        matrix = np.zeros((size, size), dtype=int)

        for node, targets in adjL.items():
            for target, weight in targets.items():
                i = node_index[node]
                j = node_index[target]
                matrix[i, j] = weight
        return matrix


    @staticmethod
    def matrix_to_adj_list(matrix: np.ndarray) -> dict:
        """
        Convert a weighted square matrix to an adjacency list.

        Parameters:
            matrix (np.ndarray): The weighted square matrix.

        Returns:
            dict: The adjacency list.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()        
            >>> matrix = np.array([[0, 1, 2], [0, 0, 3], [0, 0, 0]])
            >>> adjL = n.matrix_to_adj_list(matrix)
            >>> print(adjL)
            {'0': {'1': 1, '2': 2}, '1': {'2': 3}, '2': {}}
        """
        adj_list = {}
        size = len(matrix)
        for i in range(size):
            node = IFN.node_name(i)
            adj_list[node] = {}
            for j in range(size):
                if matrix[i][j] != 0:
                    adj_list[node][IFN.node_name(j)] = matrix[i][j]
        return adj_list


    @staticmethod
    def is_non_empty_adj_list(adjL: dict) -> bool:
        """
        Check if the adjacency list is non-empty.

        Parameters:
            adjL (dict): The adjacency list.

        Returns:
            bool: True if the adjacency list contains at least one link, otherwise False.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()        
            >>> adjL = {'a': {'b': 1}, 'b': {}}
            >>> n.is_non_empty_adj_list(adjL)
            True
        """
        return any(adjL[node] for node in adjL)


    @staticmethod
    def save_adj_list(adjL: dict, filename: str) -> None:
        """
        Save an adjacency list to a JSON file.

        Parameters:
            adjL (dict): The adjacency list.
            filename (str): The name of the file to save to.

        Example:
            >>> import IdealFlow.Network as net 
        	>>> n = net.IFN()
            >>> adjL = {'a': {'b': 1}, 'b': {}}
            >>> n.save_adj_list(adjL, 'adj_list.json')
        """
        try:
            with open(filename, 'w') as f:
                json.dump(adjL, f, indent=4)
        except IOError as e:
            raise IOError(f"Error saving adjacency list to {filename}: {e}")


    @staticmethod
    def load_adj_list(filename: str) -> dict:
        """
        Load an adjacency list from a JSON file.

        Parameters:
            filename (str): The name of the file to load from.

        Returns:
            dict: The loaded adjacency list.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()        
            >>> adjL = n.load_adj_list('adj_list.json')
            >>> print(adjL)
            {'a': {'b': 1}, 'b': {}}
        """
        try:
            with open(filename, 'r') as f:
                return json.load(f)
        except IOError as e:
            raise IOError(f"Error loading adjacency list from {filename}: {e}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in {filename}: {e}")


    @staticmethod
    def find_a_cycle(start_node: str, target_node: str, adjL: dict) -> str:
        """
        Find a cycle in the adjacency list starting from a given node.

        Parameters:
            start_node (str): The starting node.
            target_node (str): The target node to find the cycle.
            adjL (dict): The adjacency list, where keys are nodes and values are dictionaries of neighbors.

        Returns:
            str: The cycle found as a string, or None if no cycle is found.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()        
            >>> adjL = {'a': {'b': 1}, 'b': {'a': 1, 'c': 1}, 'c': {'b': 1}}
            >>> n.find_a_cycle('a', 'c', adjL)
            'abc'
        """
        visited = set()
        stack = []

        def dfs(current_node):
            visited.add(current_node)
            stack.append(current_node)
            for neighbor in adjL.get(current_node, {}):
                if neighbor == start_node and len(stack) > 1:
                    # stack.append(neighbor)
                    return ''.join(stack)
                if neighbor not in visited:
                    cycle = dfs(neighbor)
                    if cycle:
                        return cycle
            stack.pop()
            visited.remove(current_node)
            return None

        return dfs(start_node)
        
    
    @staticmethod
    def find_cycles(matrix: list) -> list:
        """
        Finds all cycles in a given adjacency matrix and returns them in canonical form.

        Parameters:
            matrix (list of list): Adjacency matrix representing the digraph.

        Returns:
            list: List of unique cycles in canonical form.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()        
            >>> matrix = [[0, 1], [1, 0]]
            >>> n.find_cycles(matrix)
            ['ab']
        """
        n = len(matrix)
        cycles = set()
        adj_list = [[] for _ in range(n)]

        for i in range(n):
            for j in range(n):
                if matrix[i][j] != 0:
                    adj_list[i].append(j)

        def canonical(cycle):
            min_idx = min(range(len(cycle)), key=lambda i: cycle[i:])
            rotated = cycle[min_idx:] + cycle[:min_idx]
            reverse = rotated[::-1]
            return min(''.join(rotated), ''.join(reverse))

        def dfs(v, start, visited, stack):
            visited[v] = True
            stack.append(v)

            for w in adj_list[v]:
                if w == start and len(stack) > 1:
                    cycle = [IFN.node_name(node) for node in stack]
                    cycles.add(canonical(cycle))
                elif not visited[w]:
                    dfs(w, start, visited, stack)

            stack.pop()
            visited[v] = False

        for i in range(n):
            visited = [False] * n
            stack = []
            dfs(i, i, visited, stack)

        return list(cycles)
    

    @staticmethod
    def find_all_permutation_cycles(matrix: list) -> set:
        """
        List all cycles in the adjacency matrix as strings, considering all permutations of cycles.

        Parameters:
            matrix (list of list of int/float): The adjacency matrix of the graph.

        Returns:
            set: A set of cycles as strings.

        Example:
            >>> import IdealFlow.Network as net
            >>> n = net.IFN()        
            >>> matrix = [[0, 1], [1, 0]]
            >>> n.find_all_permutation_cycles(matrix)
            {'ab', 'ba'}
        """
        n = len(matrix)
        cycles = set()

        def dfs(v, visited, path):
            visited[v] = True
            path.append(v)
            for i in range(n):
                if matrix[v][i] != 0:
                    if v == i:  # self-loop
                        cycles.add(chr(v + 97))
                    elif not visited[i]:
                        dfs(i, visited, path)
                    elif len(path) >= 2 and path[0] == i:
                        cycle = [chr(node + 97) for node in path]
                        cycles.add(''.join(cycle))
            path.pop()
            visited[v] = False

        for i in range(n):
            visited = [False] * n
            path = []
            dfs(i, visited, path)

        return cycles


    def str_to_num(value):
        if value.isdigit():
            return int(value)
        try:
            return float(value)
        except ValueError:
            raise ValueError("Input is not a number in str_to_num")
# END OF IFN CLASS

if __name__=='__main__':
    n = IFN()
    # adjList = {
    #     'a': {'c': 1, 'd': 2},
    #     'b': {'c': 1, 'e': 3, '#Z#': 10},
    #     'c': {'e': 5},
    #     'd': {'c': 5},
    #     'e': {'a': 3, 'b': 5},
    #     '#Z#': {'a': 10, 'b': 10}  # Cloud node connections
    # }
    # n.set_data(adjList)
    # print('adjList:',n)


    # probability, path = n.query('ebd', method='min')
    # print(f"Query Simple result: Probability: {probability}, Path: {path}") 

    # # Find minimum flow path
    # min_flow, min_path = n.min_flow_path('a', 'e')
    # print(f"Minimum flow from 'a' to 'e': {min_flow}, Path: {min_path}")

    # # Find maximum flow path
    # max_flow, max_path = n.max_flow_path('a', 'e')
    # print(f"Maximum flow from 'a' to 'e': {max_flow}, Path: {max_path}")

    # probability, path = n.query_cycle_limit('ebd', method='min', max_internal_cycle=2)
    # print(f"Query result: Probability: {probability}, Path: {path}")
    
    # probability, path = n.query_cycle_limit('ebd', method='max', max_internal_cycle=2)
    # print(f"Query result with max_internal_cycle=1: Probability: {probability}, Path: {path}")
    
    # n.show()

    
    

    
    