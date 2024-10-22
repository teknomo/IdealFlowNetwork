# -*- coding: utf-8 -*-
"""
Ideal Flow Network Core Library
# (c) 2018-2024 Kardi Teknomo
# http://people.revoledu.com/kardi/

IFN Class
This is an expandable network

version 0.14.1
last update: Oct 8,2024

Notation:
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

Metaphor:
* Path = trajectory = node sequence
* Cycle = path that have the same start and end

@author: Kardi Teknomo
"""
import numpy as np
import json
import csv
import math
import matplotlib.pyplot as plt
import networkx as nx
from fractions import Fraction
import copy
import re
from itertools import permutations, combinations
import json
import random


class IFN():

    def __init__(self, name=""):
        """
        Initialize the IdealFlowNetwork class with version information.
        """
        self.version="1.5.1"
        
        # main two variables
        self.name=name      # name of the IFN
        self.adjList={}     # adjList, weight = flow
        
        # additional object attributes
        self.numNodes=0     # number of nodes
        self.listNodes=[]   # list of nodes
        self.networkProb={} # new network weight = prob  X (Do we need this?)
        
        self.epsilon=0.000001  # precision constant
        
    
    def __repr__(self):
        """
        return string of adjList 

        example: 
        > n = IFN("test network")
        > print(n)

        see also: 
        """
        return str(self.adjList)
    
    
    def __str__(self):
        """
        return list of nodes

        see also: self.getNodes()
        """
        return str(self.adjList)
 
    
    def __len__(self):
        """
        return the number of nodes (=vertices)

        see also: self.totalNodes()
        """
        # return len(self.__adjList2listNode__(self.adjList))
        return self.numNodes
    
    
    def __iter__(self):
        """
        iterate over startNodes in adjList
        """
        return iter(self.adjList.keys())
    
   
    def __getitem__(self, link):
        """
        return weight of link [startNode,endNode]
        example usage: n[('a','b')]
        """
        (startNode, endNode)=link
        return self.__getWeightLink__(startNode,endNode)
            
    
    def __setitem__(self, link, weight):
        """
        update the weight of link [startNode,endNode]
        example usage: n[('a','b')]=3
        """
        (startNode, endNode)=link
        self.__setWeightLink__(startNode,endNode,weight)
    
    
    def setData(self,adjList):
        """
        replace the internal data structure by adjList parameter
        """
        self.adjList=adjList                               # replace the adjList
        self.listNodes=self.__adjList2listNode__(adjList)  # replace the listNodes
        self.numNodes=len(self.listNodes)                  # replace the numNodes

    

        
    '''

        NODES 
        
    '''
    
    
    def addNode(self,nodeName):
        """
        to add an isolated node

        called by: addLink() and setLinkWeight()
        """
        if nodeName not in self.listNodes:
            self.numNodes=self.numNodes+1
            self.listNodes.append(nodeName)
            self.listNodes.sort()
            self.adjList[nodeName]={}
    
    
    def deleteNode(self,nodeName):
        """
        deleting node and all connected links
        """
        if nodeName in self.listNodes:
            self.listNodes.remove(nodeName)
            self.listNodes.sort()
            self.numNodes=self.numNodes-1
            if nodeName in self.adjList:
                del self.adjList[nodeName]
            for startNode in self.adjList.keys():
                self.adjList[startNode].pop(nodeName, None)
    
    
    # 
    def getNodes(self):
        """
        return list of nodes

        see also: self.__str__()
        """
        return self.__adjList2listNode__(self.adjList)
#        return self.listNodes
    
    
    def totalNodes(self):
        """
        return total number of nodes

        see also: self.__len__()

        """
#        return len(self.__adjList2listNode__(self.adjList))
        return self.numNodes
    
    
    def nodesFlow(self):
        """
        return dictonary of {start node: sum of flow}
        
        this function is useful for node flow analysis
        for IFN this must be the same as pi
        """
        dicNode={}
        for startNode in self.adjList.keys():
            toNodes=self.outNeighbors(startNode)
            
            lst=[]
            for endNode in toNodes.keys():
                weight=toNodes[endNode]   # only use outweight
                lst.append(weight)
            dicNode[startNode]=sum(lst)
        return dicNode
    
    
    
    '''

        LINKS
        
    '''
    

    def addLink(self,startNode,endNode, weight=1):
        """
        create link if not exist, update weight with value=1 if link exists

        
        """   
        # add startNode and endNode if not exist
        if startNode not in self.listNodes:
            self.addNode(startNode)
        if endNode not in self.listNodes:
            self.addNode(endNode)
         
        if startNode in self.adjList.keys(): 
            # if startNode exists in adjList
            toNodes=self.outNeighbors(startNode)
            if endNode in toNodes.keys():
                # if endNode exist, update the link weight
                if weight>0:
                    toNodes[endNode]=toNodes[endNode]+weight
                else:
                    # if after added weight become negative
                    if toNodes[endNode]+weight<=0:
                        self.deleteLink(startNode,endNode)
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
    

    def addFirstLink(self,startNode,endNode):
        """
        shorcut to add the first link in training trajectory
        """
        self.addLink("#z#",startNode) # cloud to startNode
        self.addLink(startNode,endNode)
    
    
    def addLastLink(self,startNode,endNode):
        """
        shorcut to add the last link in training trajectory
        """
        self.addLink(startNode,endNode)
        self.addLink(endNode,"#z#") # endNode to cloud
    

    def setLinkWeight(self,startNode, endNode, weight):
        """
        set weight of a link directly
        if not exist, create the link and node
        """
        self.__setWeightLink__(startNode,endNode,weight)
    
    
    def setLinkWeightPlus1(self,startNode, endNode):
        """
        explicitly stated to set link weight plus one
        """
        self.addLink(startNode, endNode, weight=1)
    

    def get_link_flow(self,startNode, endNode):
        """
        return link flow 
        """
        if startNode not in self.listNodes or endNode not in self.listNodes:
            return np.nan
        else:
            toNodes=self.outNeighbors(startNode)
            if endNode in toNodes.keys():
                return toNodes[endNode]
            else:
                return np.nan 
            

    def deleteLink(self,startNode,endNode):
        """
        delete a link means deleting end node
        if start node is single, delete also
        """
        toNodes=self.adjList[startNode]
        del toNodes[endNode]
        toNodes=self.adjList[startNode]
        if toNodes=={}:
            del self.adjList[startNode]
    
    
    def reduceLinkFlow(self,startNode,endNode):
        """
        reduce link flow by one
        if the link flow is aleady one, delete link
        """
        self.addLink(startNode, endNode, weight=-1)
        
    
    def getLinks(self):
        """
        return list of links. A link is a list of two nodes
        """
        lst=[]
        for startNode in self.adjList.keys(): 
            toNodes=self.adjList[startNode]
            for endNode,weight in toNodes.items():
                lst.append([startNode,endNode])
        return lst
    
    
    def totalLinks(self):
        """
        return total number of links
        """
        return len(self.getLinks()) # get update
    
    
    
    '''
    
        NEIGHBORHOOD
        
    '''
    
    
    def outNeighbors(self,startNode):
        """
        return dictionary of endNodes:weight from startNode
        successors
        """
        toNodes={}
        if startNode in self.adjList:
            toNodes=self.adjList[startNode]
        return toNodes
    

    def inNeighbors(self,toNode):
        """
        return dictionary of fromNodes:weight to toNode
        predecessors
        """
        n=self.getNetworkReverse()
        return n.outNeighbors(toNode)
    
    
    def outWeight(self):
        """
        return outWeight of each node and list of nodes 
        """
        outWeigh=[]
        vertices=self.getNodes()
        for startNode in vertices:
            toNodes=self.outNeighbors(startNode)
            sumWeight=0
            for endNode,weight in toNodes.items():
                sumWeight=sumWeight+weight
            outWeigh.append(sumWeight)
        return outWeigh,vertices
    
    
    def inWeight(self):
        """
        return inWeight of each node and list of nodes
        """
        n=self.getNetworkReverse()
        return n.outWeight()
    
    
    def outDegree(self):
        """
        return outDegree of each node and list of nodes
        """
        outDeg=[]
        vertices=self.getNodes()
        for startNode in vertices:
            toNodes=self.outNeighbors(startNode)
            outDeg.append(len(toNodes))
        return outDeg,vertices
    
    
    def inDegree(self):
        """
        return inDegree of each node and list of nodes
        """
        n=self.getNetworkReverse()
        return n.outDegree()
    
    
    '''
    
        NETWORK INDICES
        
    '''
    

    def density(self):
        """
        return the density of a graph
        https://www.python-course.eu/graphs_python.php
        The graph density is defined as the ratio of 
        the number of edges of a given graph, and 
        the total number of edges, the graph could have. 
        It measures how close a given graph is to a complete graph.
        The maximal density is 1, if a graph is complete.
        """
        V = self.totalNodes()
        E = self.totalLinks()
        return 2.0 * E / (V *(V - 1))
    

    def diameter(self):
        """
        return diameter of network
        https://www.python-course.eu/graphs_python.php
        The diameter d of a graph is defined as 
        the maximum eccentricity of any vertex in the graph. 
        The diameter is the length of the shortest path 
        between the most distanced nodes. 
        To determine the diameter of a graph, 
        first find the shortest path between each pair of vertices. 
        The greatest length of any of these paths is the diameter of the graph.
        """
        v = self.getNodes() 
        pairs = [ (v[i],v[j]) for i in range(len(v)-1) for j in range(i+1, len(v))]
        smallestPaths = []
        for (s,e) in pairs:
            paths = self.findAllPaths(s,e)
            if paths!=[]:
                smallest = sorted(paths, key=len)[0]
                smallestPaths.append(smallest)
        smallestPaths.sort(key=len)
        # longest path is at the end of list, 
        # i.e. diameter corresponds to the length of this path
        diameter = len(smallestPaths[-1]) - 1
        return diameter
    
    
    def totalFlow(self):
        """
        return scalar total flow
        """
        kappa=0
        for startNode in self.adjList.keys(): 
            toNodes=self.outNeighbors(startNode)
            for endNode,weight in toNodes.items():
                kappa=kappa+weight
        return kappa


    # 
    def stdevFlow(self):
        """
        return standard deviation of flow
        """
        m=self.totalLinks()
        avg=self.totalFlow()/m
        std=0
        count=0
        for startNode in self.adjList.keys(): 
            toNodes=self.outNeighbors(startNode)
            for endNode,weight in toNodes.items():
                std=std+(weight-avg)**2
                count=count+1
        if std>0 and count>0:
            std=math.sqrt(std)/count
        else:
            std=0
        return std
    
    
    def covFlow(self):
        """
        return coef. of variatio of flow
        """
        return self.stdevFlow()/(self.totalFlow()/self.totalLinks())
    
    
    def maxLinkFlow(self):
        """
        return max flow in the network
        """
        lst=[]
        for key, val in self.adjList.items():
            if val!={}:
                lst.append(max(val.values()))
        return max(lst)

    
    def getNetworkEntropy(self):
        """
        return scalar network entropy
        """
        F,listNode=self.getMatrix()
        S=self.idealFlow2stochastic(F)
        return self.networkEntropy(S)
    
     

    '''
    
        STOCHASTIC
        
    '''
        
    
    def getStochastic(self):
        """
        return adjList with value = link probability out of each node
        """
        F,listNode=self.getMatrix()
        S=self.idealFlow2stochastic(F)
        n=IFN()
        n.applyMatrix(S,listNode)
        return n.adjList 
        
    
    def getNetworkProbability(self):
        """
        return adjList with value = link probability (out of kappa)
        """
        self.__updateNetworkProbability__()
        return self.networkProb#.adjList



    '''
        TRAJECTORY = PATH
        
    '''
    
    
    def findPath(self, startNode, endNode, path=[]):
        """
        return a path from startNode to endNode

        the same as backtracking but using DFS recursive
        """
        path = path + [startNode]
        if startNode == endNode:
            return path
        if startNode not in self.adjList:
            return None
        for node in self.adjList[startNode]:
            if node not in path:
                extended_path = self.findPath(node, endNode,path)
                if extended_path: 
                    return extended_path
        return []
    

    def findAllPaths(self, startNode, endNode, path=[]):
        """
        return list of all possible paths from startNode to endNode
        """
        path = path + [startNode]
        if startNode == endNode:
            return [path]
        if startNode not in self.adjList:
            return []
        paths = []
        for node in self.adjList[startNode]:
            if node not in path:
                extended_paths = self.findAllPaths(node,endNode,path)
                for p in extended_paths: 
                    paths.append(p)
        return paths
    
    
    def shortestPath(self,startNode,endNode):
        """
        return min weight path (node sequence) between startNode & endNode
        based on Dijkstra

        note: this is min number of links not min weight!!!
        """
        paths = self.findAllPaths(startNode,endNode)
        if paths!=[]:
            shortest = sorted(paths, key=len)[0]
        else:
            shortest = []
        return shortest
    
    
    def allShortestPath(self):
        """
        return matrix of all shortest path
        based on Floyd-Warshall algorithm

        note: this is min weight path, not min number of links
        """
        # prepare the dist matrix with inf to replace 0
        m,listNode=self.getMatrix()
        n=len(listNode)
        m=self.__matrixReplaceValue__(m,0,math.inf) # replace zero with math.inf
        
        for k in range(n):
            d = [list(row) for row in m] # make a copy of distance matrix
            for i in range(n):
                for j in range(n):
                    # Choose if the k vertex can work as a path with shorter distance
                    d[i][j] = min(m[i][j], m[i][k] + m[k][j])
            m=d
        return m,listNode
    
    
    def isPath(self,nodeSequence):
        """
        return True if nodeSequence is path
        """
        for idx,node1 in enumerate(nodeSequence[:-1]):
            node2=nodeSequence[idx+1]
            weight=self.__getWeightLink__(node1,node2)
            if weight==0:
                return False
        return True
    
     
    def setPath(self, nodeSequence):
        """
        set a trajectory into the network by 
            updating the link weight +1 if exist
            create link with weight=1, if not exist
        
        note: deliberately not test isPath here
        """
        for idx,startNode in enumerate(nodeSequence[:-1]):
            endNode=nodeSequence[idx+1]
            self.setLinkWeightPlus1(startNode, endNode)
            

    def isCycle(self,path):
        """
        return true if node sequence path is a cycle
        """
        return self.isPath(path) and path[0]==path[-1]
    

    def cycleLength(self,cycle):
        """
        return number of edges if cycle, else return 0
        cycle is defined as node sequence with start=end
        """
        if self.isCycle(cycle):
            return len(cycle)-1
        else:
            return 0
     
    
    def cycleWeight(self,cycle):
        """
        return sum of weight in a cycle, else return 0
        cycle is defined as node sequence with start=end
        """
        if self.isCycle(cycle):
            return self.pathWeight(cycle)
        else:
            return 0
        
    
    def pathLength(self,path):
        """
        return number of edges if path, else return 0
        path is defined as node sequence
        """
        if self.isPath(path):
            return len(path)-1
        else:
            return 0
    
    
    def pathDistance(self,startNode,endNode):
        """
        return sum weight of the shortest path between startNode & endNode
        """
        shortest=self.shortestPath(startNode,endNode)
        return self.pathWeight(shortest)
    
    
    def pathWeight(self,path):
        """
        return sum of weight in path
        path is defined as node sequence

        note: deliberately not test isPath here
        """
        sum=0
        for idx,node1 in enumerate(path[:-1]):
            node2=path[idx+1]
            weight=self.__getWeightLink__(node1,node2)
            sum=sum+weight
        return sum    
    
    
    def randomWalk(self,startNode,length=1):
        """
        return list of nodes passed by random walk from startNode
        """
        result=[]
        currentNode=startNode
        result.append(currentNode)
        for n in range(length):
            toNodes=self.outNeighbors(currentNode)
            if toNodes !={}:
                listNodes=[]
                listWeight=[]
                for k,w in toNodes.items():
                    listNodes.append(k)
                    listWeight.append(w)
            probs = [x/sum(listWeight) for x in listWeight]
            currentNode=np.random.choice(listNodes,p=probs)
            result.append(currentNode)
        return result
    
    
    def randomCycle(self,startEndNode):
        """
        return list of nodes passed by random cycle from startNode
        """
        result=[]
        currentNode=startEndNode
        result.append(currentNode)
        while True:
            toNodes=self.outNeighbors(currentNode)
            listNodes=[]
            listWeight=[]
            if toNodes !={}:
                for k,w in toNodes.items():
                    listNodes.append(k)
                    listWeight.append(w)
                probs = [x/sum(listWeight) for x in listWeight]
                currentNode=np.random.choice(listNodes,p=probs)
                result.append(currentNode)
            if startEndNode==currentNode:
                break
        return result
    
    
    def getPathProbability(self,nodeSequence,isUpdateFirst=False):
        """
        return avg probability of a trajectory and 
        return number of links until it reaches zero flow link

        if trajectory has no path, avg prob = 0
        if network probability was not computed, it will update 
        """
        if nodeSequence==[]:
            return 0,0
        if self.networkProb=={} or isUpdateFirst==True:
            self.__updateNetworkProbability__() # update self.networkProb
        sumProb=0
        numLink=0
        for idx,startNode in enumerate(nodeSequence[:-1]):
            endNode=nodeSequence[idx+1]
            adjList=self.networkProb
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
    
    
    def getPathEntropy(self,nodeSequence,isUpdateFirst=False):
        """
        return entropy of a trajectory
        if network probability was not computed, it will update
        if link probability is zero, it would be ignored from entropy computation
        """
        if nodeSequence==[]:
            return 0
        if self.networkProb=={} or isUpdateFirst==True:
            self.__updateNetworkProbability__() # update self.networkProb
        sumEntropy=0
        numLink=0
        for idx,startNode in enumerate(nodeSequence[:-1]):
            endNode=nodeSequence[idx+1]
            adjList=self.networkProb#.adjList
            linkProb=self.__getLinkWeight__(startNode,endNode,adjList)
            numLink=numLink+1
            if linkProb>0:
                sumEntropy=sumEntropy-linkProb*math.log(linkProb,2)
        if numLink>0:
            avgEntropy=sumEntropy/numLink
        else:
            avgEntropy=0
        return avgEntropy
    
    
    '''
    
        SEARCH 
        
    '''
    

    def dfs(self,startNode): 
        """
        return node sequence of nodes (not path) visited in dfs manner
        """
        vertices=self.getNodes()
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
                toNodes=self.outNeighbors(currentNode)
                for node in toNodes:
                    if (not visited[node]):  
                        stack.append(node)
        return dfsPath

    
    def dfsUntil(self,startNode, endNode): 
        """
        return node sequence of nodes (not path) traversed in dfs manner
        until it reaches endNode
        """
        dfsPath=[]
        vertices=self.getNodes()
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
                toNodes=self.outNeighbors(currentNode)
                for node in toNodes:
                    
                    if (not visited[node]):  
                        stack.append(node)
        return dfsPath

   
    def bfs(self,startNode): 
        """
        return node sequence of nodes (not path) traversed in bfs manner

        """
        bfsPath=[]
        vertices=self.getNodes()
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
            toNodes=self.outNeighbors(currentNode)
            for node in toNodes:
                if (not visited[node]):  
                    queue.append(node)
                    visited[node] = True 
        return bfsPath


    def bfsUntil(self,startNode,endNode): 
        """
        return node sequence of nodes (not path) traversed in bfs manner
        until it reaches endNode
        """
        bfsPath=[]
        vertices=self.getNodes()
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
            toNodes=self.outNeighbors(currentNode)
            for node in toNodes:
                if (not visited[node]):  
                    queue.append(node)
                    visited[node] = True
                    
        return bfsPath
    

    def backtracking(self,startNode, endNode): 
        """
        return node sequence PATH from startNode to endNode
        based on DFS
        """
        dfsPath=[]
        vertices=self.getNodes()
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
                toNodes=self.outNeighbors(currentNode)
                if toNodes=={}:
                    # if no unvisited outneighbor then
                    # backtrack due to leaf node
                    while dfsPath!=[] and dfsPath[-1]==currentNode:
                        currentNode = predecessor[currentNode]
                        dfsPath.pop()
                        toNodes=self.outNeighbors(currentNode)
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
                            toNodes=self.outNeighbors(currentNode)
                            isFound=False
                            for node in toNodes:
                                if node in stack and (not visited[node]):
                                    isFound=True
                                    break
                            if isFound==True:
                                break 
        return dfsPath
    
    

    '''
    
        DATA SCIENCE RELATED
        
    '''

    def unlearn(self,trajectory):
        """
        unassigned trajectory from network
        put weight=-1 along the path of trajectory
        trajectory is a node sequence
        """
        for idx,startNode in enumerate(trajectory[:-1]):
            endNode=trajectory[idx+1]
            self.addLink(startNode, endNode, weight=-1)
            

    def assign(self,trajectory):
        """
        assign trajectory to network
        trajectory is a node sequence

        alias of setPath()
        """
        self.setPath(trajectory)
    
    
    def generate(self,startEndNode="#z#"):
        """
        generate random cycle from cloud node to cloud node
        alias of randomCycle()
        """
        return self.randomCycle(startEndNode)
    

    def match(self,trajectory, dicIFNs):
        """
        return the name of IFN from the list of IFNs that 
        has the max trajectory entropy
        and percentage of maxEnt/sum of entropy
        """
        dicEntropy={}
        lst=[]
        for name,n in dicIFNs.items():
            # trajectory entropy of each IFN
            h=n.getPathEntropy(trajectory) 
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
            
    
    def trajectory2Links(self,trajectory):
        """
        given a list of nodes, provide links

        e.g. ['a','b','c','d'] ==> [['a','b'],['b','c'],['c','d']]
        generic: the node sequence is not necessarily a path in the network
        """
        lst=[]
        for idx,startNode in enumerate(trajectory[:-1]):
            endNode=trajectory[idx+1]
            lst.append([startNode,endNode])
        return lst
    
    
    def linkCombination(self,trajectory):
        """
        given a list of nodes, provide link combination (one way)
        e.g. ['a','b','c','d'] ==> [['a','b'],['a','c'],['a','d'],['b','c'],['b','d'],['c','d']]
        generic: the node sequence is not necessarily a path in the network
        """
        lst=[]
        for idx,node1 in enumerate(trajectory[:-1]):
            for idx2,node2 in enumerate(trajectory[idx+1:]):
                lst.append([node1,node2])
        return lst
    

    def linkPermutation(self,trajectory):
        """
        given a list of nodes, provide link permutation (two ways)

        e.g. ['a','b','c','d'] ==> 
        ==> [['a','b'],['a','c'],['a','d'],['b','c'],['b','d'],['c','d'],
             ['b','a'],['c','a'],['d','a'],['c','b'],['d','b'],['d','c']]
        generic: the node sequence is not necessarily a path in the network

        """
        lst=[]
        for idx,node1 in enumerate(trajectory):
            for idx2,node2 in enumerate(trajectory):
                if len(trajectory)==1 or idx!=idx2:
                    lst.append([node1,node2])
        return lst
    
   
    def associationTraining(self,trajectory,net):
        """
        training IFN for association based on one trajectory data
        assume to have two ways link permutation

        """
        if net=={}:
            net=IFN()
        g=net.completeGraph(trajectory)
        net=net.overlay(g, net)
        return net
    
    
    def associationPredictTrajectory(self,trajectory,net):
        """
        given a trajectory, and IFN
        predict the association of itemset
        based on direct link from trajectory complete graph to the IFN that is not in the complete graph
    
        return 
            prediction = sorted dictionary of item: flow
            supp=count of flow in trajectory items
            conf=count of flow in all direct links
        """
        nodeFlow=net.nodesFlow()
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
    
    
    def associationPredictActorNet(self,netActor,netSystem):
        """
        """
        nodeFlow=netActor.nodesFlow()
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
        
    
    def orderMarkovLower(self,trajSuper):
        """
        Convert trajectory of high order Markov into first order Markov
        Agreement: 
            separator between node in supernode is '|'
            cloud node is '#z#' and always first order
        input:
            trajSuper = listtrajectory ctory of super node of K order
        output:
            traj = list of node sequence first order in hash code
        """
        lstCloud=self.__findElementInList__('#z#', trajSuper)
        trajS=trajSuper
        try:
            while True:
               trajS.remove('#z#')
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
            traj.insert(0,"#z#")
            traj.insert(len(traj),"#z#")
        return traj

    
    def orderMarkovHigher(self,trajectory,order):
        """
        Convert trajectory of first order Markov into higher order Markov
        Agreement: 
            separator between node in supernode is '|'
            cloud node is '#z#' and always first order
        input:
            trajectory=list of node sequence first order in hash code
            order = higher order Markov
        output:
            trajSuper = listtrajectory ctory of super node of K order
        """
        delim='|'
        q=len(trajectory)
        lstCloud=self.__findElementInList__('#z#', trajectory)
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
            if '#z#' in superNode:
                superNode='#z#'
            trajSuper.append(superNode)
        return trajSuper
    

    def toMarkovOrder(self,trajectory,toOrder):
        """
        conversion from trajectory of any order to any order
        """
        traj=self.orderMarkovLower(trajectory) # put to first order first
        trajSuper=self.orderMarkovHigher(traj,toOrder) # before going higher markov order
        return trajSuper  
    


    '''
    
        TESTING NETWORK
        
    '''
    

    def isEqualNetwork(self,net1,net2):
        """
        return true if net1==net2
        """
        return net1.adjList==net2.adjList        
    
    
    def isEquivalentIFN(self,ifn):
        """
        return true if internal IFN is equivalent to parameter IFN

        test based on coef. of variation of flow
        """
        cov1=self.covFlow()
        cov2=ifn.covFlow()
        if abs(cov1-cov2)<self.epsilon:
            return True
        else:
            return False
    
    # 
    def isReachable(self,startNode,endNode): 
        """
        return true if endNode is reachable from startNode
        """
        vertices=self.getNodes()
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
            toNodes=self.outNeighbors(startNode)            
            for node in toNodes:
                idx=vertices.index(node)
                if (not visited[idx]):  
                    queue.append(node)
                    visited[idx] = True 
        return False
    
    
    def isContainCycle(self):
        """
        returns true if there is a cycle  in internal IFN
        """ 
        vertices=self.getNodes()
        n=len(vertices)
        in_degree=[0]*n # indegrees of all nodes
      
        # Traverse adjacency lists to fill indegrees of  
        # vertices. This step takes O(V+E) time 
        for startNode in vertices:
            toNodes=self.outNeighbors(startNode)
            for node in toNodes:
                idx=vertices.index(node)
                in_degree[idx]+=1
          
        # enqueue all vertices with indegree 0 
        queue=[] 
        for i in range(len(in_degree)): 
            if in_degree[i]==0: 
                v=vertices[i]
                queue.append(v) 
          
        cnt=0 # Initialize count of visited vertices 
      
        # One by one dequeue vertices from queue and enqueue  
        # adjacents if indegree of adjacent becomes 0  
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
    
    
    def isAcyclic(self):
        """
        return True if the internal network contain no cycle
        """
        return self.isContainCycle()==False
    
    
    def isConnected(self):
        """
        return True if the internal network is connected
        """ 
        n=self.toGraph()      # make both directions
        vertices=n.getNodes()
        startNode=vertices[0]
        nodeSequence=n.dfs(startNode)
        if len(nodeSequence)!=len(vertices):
            return False
        # all nodes is visited in any direction
        return True
    

    def isStronglyConnected(self):
        """
        return True if the internal network is strongly connected
        """
        vertices=self.getNodes()
        
        # If BFS doesn't visit all nodes it is not strongly connected
        for startNode in vertices:
            nodeSequence=self.bfs(startNode)
            if len(nodeSequence)!=len(vertices):
                return False
        # reverse the links
        net=self.getNetworkReverse()
        # If BFS of reverse network doesn't visit all nodes it is not strongly connected
        for startNode in vertices:
            nodeSequence=net.bfs(startNode)
            if len(nodeSequence)!=len(vertices):
                return False
        
        return True # otherwise strongly connected
    
    
    def isPremagicNetwork(self):
        """
        return true if inWeight~=outWeight for all nodes
        """
        inWeigh,vertices=self.inWeight()
        outWeigh,vertices=self.outWeight()
        for i,v in enumerate(vertices):
            if abs(inWeigh[i]-outWeigh[i])>self.epsilon:
                return False
        return True
    
    # 
    def isIdealFlow(self):
        """
        return True if the weights are premagic and strongly connected network
        """
        if self.isStronglyConnected() and self.isPremagicNetwork():
            return True
        else:
            return False
        
    
    def isEulerianCycle(self):
        """
        return True if the internal network is Eulerian Cycle
        """ 
        # Check if all non-zero degree vertices are connected 
        if self.isStronglyConnected() == False: 
            return False
  
        # Check if in degree and out degree of every vertex is same 
        inDeg,vertices=self.inDegree()
        outDeg,vertices=self.outDegree()
        for i,v in enumerate(vertices):
            if inDeg[i]!=outDeg[i]:
                return False
  
        return True
    

    def isBipartite(self):
        """
        return True if the internal network is bipartite 

        based on https://www.geeksforgeeks.org/bipartite-graph/
        """
        M,nodes=self.getMatrix()
        V=self.totalNodes()
        color = [-1] * V  
              
        #start is vertex 0  
        pos = 0 
        # two colors 1 and 0  
        retVal=self.__colorGraph(M, color, pos, 1)
#        if retVal: print('color=',color,'\n')
        return retVal
    
    

    '''
        NETWORK OPERATIONS
        
    '''
    @staticmethod
    def union(net1,net2):
        """
        return the union of two network inputs
        """
        n=IFN()  # create new network
        
        for startNode1 in net1.adjList:
            toNodes1=net1.adjList[startNode1]
            for endNode1,weight1 in toNodes1.items():
                n.addLink(startNode1,endNode1,weight1)
        for startNode2 in net2.adjList:        
            toNodes2=net2.adjList[startNode2]                
            for endNode2,weight2 in toNodes2.items():
                n.addLink(startNode2,endNode2,weight2)
        n1=IFN()
        n1=n1.intersect(net1, net2)
        n1=n.difference(n, n1)            
        return n1
    
    @staticmethod
    def overlay(net2,net1):
        """
        put net1 into net2
        set union (net2+net1)
        return updated weights of network2 (or expanded network2 by network1)
        input net1=IFN (smaller)
              net2=IFN (base - usually larger)
        """
        for startNode in net1:
            toNodes=net1.adjList[startNode]
            for endNode,weight in toNodes.items():
                net2.addLink(startNode,endNode,weight)
        return net2
        
    @staticmethod
    def difference(net2,net1):
        """
        reduce link flow of net2 based on net1
        = set difference (net2-net1)
        
        return updated weights of network2 (or shrinked network2 by network1)
        input net1=IFN (smaller)
              net2=IFN (base - usually larger) 
        """
        n=net2.duplicate()
        for startNode in net1:
            toNodes=net1.adjList[startNode]
            for endNode, flow in toNodes.items():
                n.addLink(startNode,endNode,weight=-flow)
        return n
    
    @staticmethod
    def complement(net):
        """
        return network diference complete graph - net1
        """
        n=IFN()  # create new network
        U=n.universe(net)
        n=n.difference(U,net)
        return n
    
    @staticmethod
    def intersect(net1,net2):
        """
        return intersection of two networks 
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
                            n.addLink(startNode,endNode1,weight)
        return n

    @staticmethod
    def universe(net):
        """
        return universe of the given network (= complete digraph)
        """
        vertices=net.getNodes()
        m=net.maxLinkFlow()
        U=net.completeGraph(vertices,weight=m)
        return U
    

    '''
        NEW NETWORKS
        
    '''
    
    @staticmethod
    def completeGraph(trajectory,weight=1):
        """
        return two ways complete graph weight 1 from a trajectory list
        if trajectory has only one item, create a node
        a complete graph weight 1 is always IFN
        """
        n=IFN()  # create new network
        for idx1,startNode in enumerate(trajectory):
            for idx2,endNode in enumerate(trajectory):
                if len(trajectory)==1:
                    n.addNode(trajectory[0])
                elif idx1!=idx2:
                    n.addLink(startNode, endNode,weight)
        return n
    
    
    def duplicate(self):
        """
        return duplicate this network (not the same reference)
        """
        n=IFN()  # create new network
        # main copy
        n.name=self.name
        n.adjList=copy.deepcopy(self.adjList)
        # also copy
        n.listNodes=copy.deepcopy(self.listNodes)
        n.numNodes=self.numNodes        
        return n
    
    
    def toGraph(self):
        """
        from digraph to graph
        return new network of graph counterpart of the digraph
        the adjacency matrix is symmetric
        the link weights are adjusted to all 1
        """
        n=IFN()  # create new network
        nR=self.getNetworkReverse()
        
        # copy from current
        for startNode in self.adjList.keys(): 
            toNodes=self.outNeighbors(startNode)
            for endNode,weight in toNodes.items():
                n.addLink(endNode,startNode,1)
        # copy also from reverse
        for startNode in nR.adjList.keys(): 
            toNodes=nR.outNeighbors(startNode)
            for endNode,weight in toNodes.items():
                n.addLink(endNode,startNode,1)
        return n
    
    
    def getNetworkReverse(self):
        """
        return network with link reverse direction
        """
        n=IFN()
        for startNode in self.adjList.keys(): 
            toNodes=self.outNeighbors(startNode)
            for endNode,weight in toNodes.items():
                n.addLink(endNode,startNode,weight)
        n.reindex()
        return n
    
    
    def networkNoCloud(self):
        """
        copy me and delete cloud node
        """
        net=self.duplicate()
        net.deleteNode("#z#")
        return net
    


    '''
    
        MATRICES
        
    '''

    def getMatrix(self):
        """
        return adjacency matrix of the network and list of nodes
        """
        return self.__adjList2Matrix__(self.adjList)
    
    
    def applyMatrix(self,M,listNode=[]):
        """
        replace self.adjList by external matrix

        this is useful if we use matrices in computation and
        want to put the matrix into network
        """
        if listNode==[]:
            # set up default list node if not specified
            size=np.array(M).shape
            mC=size[0]
            listNode=[self.__num_to_excel_col__(x) for x in range(1,mC+1)]
        self.adjList=self.__matrix2AdjList__(np.array(M),listNode)
        self.listNodes=listNode
    
    @staticmethod
    def binarizedMatrix(M):
        """
        return (0,1) matrix of M
        """
        return [[int(bool(x)) for x in l] for l in M]

    '''
    
        NEW IN THIS VERSION
        
    '''
    @staticmethod
    def capacity2adj(C):
        '''
        convert capacity matrix to adjacency matrix
        '''
        return(np.asarray(C)>0).astype(int) # get adjacency matrix structure

    @staticmethod
    def capacity2stochastic_proportional(C):
        '''
        convert capacity matrix into stochastic matrix
        S=C./(sR*ones(1,n))
        '''
        s=np.apply_along_axis(np.sum, axis=1, arr=C)
        return C/s[:,np.newaxis]

    @staticmethod
    def adj2stochastic(A):
        '''
        convert adjacency matrix to stochastic matrix 
        of equal outflow distribution
        '''
        v=np.sum(A,axis=1)           # node out degree
        D=np.diag(v)                 # degree matrix
        return np.dot(np.linalg.inv(D),A) # ideal flow of equal outflow distribution

    @staticmethod
    def idealFlow2stochastic(F):
        '''
        convert ideal flow matrix into Markov stochastic matrix
        ''' 
        s=np.apply_along_axis(np.sum, axis=1, arr=F)
        return F/s[:,np.newaxis]

    # def markov(self, S, kappa=1):
    #     """
        
    #     """
    #     S = np.array(S)
    #     n = S.shape[0]
    #     I = np.eye(n)
    #     j = np.ones(n)
    #     X = np.vstack([S.T - I, j])
    #     Xp = np.linalg.pinv(X)
    #     y = np.zeros(n + 1)
    #     y[-1] = kappa
    #     return np.dot(Xp, y).tolist()
    @staticmethod
    def Markov(S,kappa=1):
        '''
        Compute the Markov chain for a given stochastic matrix.
        convert stochastic matrix into steady state Markov vector
        kappa is the total of Markov vector
        
        Parameters:
        S (list of list of float): The stochastic matrix.
        kappa (float): The kappa parameter.

        Returns:
        np array of float: The Markov chain.
        
        previous version: steadyStateMC()
        '''
        if not isinstance(S, np.ndarray):
            S = np.array(S)
        [m,n]=S.shape
        if m==n:
            I=np.eye(n)
            j=np.ones((1,n))
            X=np.concatenate((np.subtract(S.T,I), j), axis=0) # vstack
            Xp=np.linalg.pinv(X)      # Moore-Penrose inverse
            y=np.zeros((m+1,1),float)
            y[m]=kappa
            pi=np.dot(Xp,y)
            return pi

    @staticmethod
    def idealFlow(S,pi):
        '''
        return ideal flow matrix
        based on stochastic matrix and Markov vector
        '''
        return S*pi
        # [m,n]=S.shape
        # jt=np.ones((1,n))
        # B=pi.dot(jt)
        # return np.multiply(B,S)

    @staticmethod
    def adj2IdealFlow(A,kappa=1):
        '''
        convert adjacency matrix into ideal flow matrix 
        of equal distribution of outflow 
        kappa is the total flow
        '''
        S=IFN.adj2stochastic(A)
        pi=IFN.Markov(S,kappa)
        return IFN.idealFlow(S,pi)
        
    @staticmethod
    def capacity2IdealFlow(C,kappa=1):
        '''
        convert capacity matrix into ideal flow matrix
        kappa is the total flow
        '''
        S=IFN.capacity2stochastic_proportional(C)
        pi=IFN.Markov(S,kappa)
        return IFN.idealFlow(S,pi)

    @staticmethod
    def congestion(F,C):
        """
        Compute congestion matrix from flow and capacity matrices.
        
        congestion matrix, which is element wise
        division of flow/capacity, except zero remain zero
        
        Parameters:
        F (list of list of int/float or 2D np.array): The flow matrix.
        C (list of list of int/float or 2D np.array): The capacity matrix.

        Returns:
        2D np.array of float: The congestion matrix.
        """
        return IFN.hadamardDivision(F,C)

    @staticmethod
    def sumOfRow(M):
        '''
        return vector sum of rows
        '''
        [m,n]=M.shape
        j=np.ones((m,1))
        return np.dot(M,j)    

    @staticmethod
    def sumOfCol(M):
        '''
        return row vector sum of columns
        '''
        [m,n]=M.shape
        j=np.ones((1,n))
        return np.dot(j,M)

    @staticmethod
    def isSquareMatrix(M):
        '''
        return True if M is a square matrix
        '''
        [m,n]=np.array(M).shape
        if m==n:
            return True
        else:
            return False

    @staticmethod
    def isNonNegativeMatrix(M):
        '''
        return True of M is a non-negative matrix
        '''
        if np.any(np.array(M)<0):
            return False
        else:
            return True

    @staticmethod
    def isPositiveMatrix(M):
        '''
        return True of M is a positive matrix
        '''
        if np.any(np.array(M)<=0):
            return False
        else:
            return True
            
    @staticmethod
    def isPremagicMatrix(M):
        '''
        return True if M is premagic matrix (must be 2D np.array), 
        that is the sum of rows is equal to the sum of columns
        '''
        M=np.array(M)
        (n,m)=M.shape
        j=np.ones((n,1))
        sR=np.dot(M,j)
        sC=np.dot(M.transpose(),j)
        retVal=np.allclose(sR,sC)
        return retVal
        
    @staticmethod
    def isIrreducible(M):
        '''
        return True if M is irreducible matrix 
        '''
        M=np.array(M)
        if IFN.isSquareMatrix(M) and IFN.isNonNegativeMatrix(M):
            [m,n]=M.shape
            I=np.eye(n)
            Q=np.linalg.matrix_power(np.add(I,M),n-1) # Q=(I+M)^(n-1)
            return IFN.isPositiveMatrix(Q)
        else:
            return False

    @staticmethod
    def isIdealFlowMatrix(M):
        '''
        return True if M is an ideal flow matrix
        '''
        if IFN.isNonNegativeMatrix(M) and IFN.isIrreducible(M) and IFN.isPremagicMatrix(M):
            return True
        else:
            return False

    @staticmethod
    def equivalentIFN(F,scaling):
        '''
        return scaled ideal flow matrix
        input:
        F = ideal flow matrix
        scaling = global scaling value
        '''
        F1=F*scaling
        return F1
    
    @staticmethod
    def globalScaling(F,scalingType='min',val=1):
        '''
        return scaling factor to ideal flow matrix
        to get equivalentIFN
        input:
        F = ideal flow matrix
        scalingType = {'min','max','sum','int'}
        val = value of the min, max, or sum
        'int' means basis IFN (minimum integer)
        '''
        f=np.ravel(F[np.nonzero(F)]) # list of non-zero values in F
        # print('f',f)
        if scalingType=='min' and min(f)>0:
            opt=min(f)
            scaling=val/opt
        elif scalingType=='max' and max(f)>0:
            scaling=val/max(f)
        elif scalingType=='sum' and sum(f)>0:
            scaling=val/sum(f)
        elif scalingType=='int':
            denomSet=set()
            for g in f:
                h=Fraction(g).limit_denominator(1000000000)
                denomSet.add(h.denominator)
            scaling=1
            for d in denomSet:
                scaling=IFN.lcm(scaling,d)
        else:
            raise ValueError("unknown scalingType")
        return scaling

    @staticmethod
    def minIrreducible(k):
        '''
        return min irreducible matrix size n by n
        '''
        A=np.zeros((k,k),dtype= np.int8)
        for r in range(k-1):
            c=r+1
            A[r,c]=1
        A[k-1,0]=1
        return A

    @staticmethod
    def addRandomOnes(A,m=6):
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

    @staticmethod
    def randIrreducible(k=5,m=8):
        '''
        return random irreducible matrix size n by n
        input:
        k = total number of nodes
        m = total number of links >=n
        '''
        A=IFN.minIrreducible(k)    # create min irreducible matrix
        A1=IFN.addRandomOnes(A,m)  # add random 1 up to m
        P=IFN.randPermutationEye(k)  # random permutation of identity matrix
        A2=np.dot(np.dot(P,A1),P.transpose()) # B=P.A.P' shufffle irreducible matrix to remain irreducible
        return A2

    @staticmethod
    def randPermutationEye(n=5):
        '''
        return random permutation matrix of identity matrix size n
        '''
        eye =np.identity(n)
        np.random.shuffle(eye)
        return eye

    @staticmethod
    def coefVarFlow(F):
        '''
        return coeficient variation of the Flow matrix
        '''
        mean=np.mean(F)
        std=np.std(F)
        return mean/std

    @staticmethod
    def networkEntropy(S):
        '''
        return the value of network entropy
        '''
        s=S[np.nonzero(S)]
        return np.sum(np.multiply(-s,np.log(s)),axis=None)

    @staticmethod
    def entropyRatio(S):
        '''
        return network entropy ratio
        '''
        h1=IFN.networkEntropy(S)
        A=(S>0).astype(int) # get adjacency matrix structure
        T=IFN.adj2stochastic(A)
        h0=IFN.networkEntropy(T)
        if h0>0:
            return h1/h0
        else:
            return 0

        
    
    '''
    
        UTILITIES
    
    '''
    
    
    def reindex(self):
        """
        sort the nodes, standardize the adjacency list
        be careful: take a long time for big data
        """
        m,listNode=self.__adjList2Matrix__(self.adjList)
        self.adjList=self.__matrix2AdjList__(m,listNode)
    
    
    def save(self,fileName):
        """
        save self.dicNode and self.adjList to a file
        """
        with open(fileName,'w') as fp:
            json.dump(self.adjList,fp,sort_keys=True,indent=4)
    
    
    def load(self,fileName):
        """
        load self.dicNode and self.adjList from a file
        """
        with open(fileName,'r') as fp:
            self.adjList=json.load(fp)
    
    @staticmethod
    def readCSV(fName): 
        """
        return 2D array from CSV file
        """   
        data=[]
        with open(fName, newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                col=list(map(str.strip,row))
                data.append(col)
        return data
    
    @staticmethod
    def flatten(nDArray):
        """
        flatten nD numpy array
        """
        return list(np.concatenate(nDArray).flat) 
    

    '''
        return least common multiple of two large numbers
    '''
    @staticmethod
    def lcm(a,b):
        return a*b // math.gcd(a,b)
    
    @staticmethod
    def lcmList(lst):
        '''
        return least common multiple from a list of integer numbers
        '''
        a = lst[0]
        for b in lst[1:]:
            a = IFN.lcm(a,b)
        return a

    @staticmethod
    def hadamardDivision(a, b):
        """ elementwise division by ignore / 0, https://stackoverflow.com/questions/26248654/how-to-return-0-with-divide-by-zero """
        with np.errstate(divide='ignore', invalid='ignore'):
            c = np.true_divide( a, b )
            c[ ~ np.isfinite( c )] = 0  # -inf inf NaN
        return c
    
    
    '''
    
     show the network
    
    '''
    
    
    def show(self,mNode=None,arrThreshold=None,routes=None,layout=None):
        """
        show network    
        """
        vertices=self.getNodes()
        totalNodes=len(vertices)
        plt.figure()
        G = nx.DiGraph()
#        G=nx.MultiDiGraph()
        G.add_nodes_from(vertices)
        for n1,dic in self.adjList.items():
            for n2,cost in dic.items():
                G.add_edge(n1, n2, weight=int(round(cost,2)))
        if mNode is None:
            nodeIds=range(0,totalNodes)
            mRn=len(nodeIds)
        else:
            mNode=np.array(mNode)
            mRn=len(mNode)
            for r in range(mRn):
                nodeID=mNode[r,0]
                x=float(mNode[r,1])
                y=float(mNode[r,2])
                G.add_node(nodeID,pos=(x,y))
                
        if mNode is None:
            # if node position is set automatically
            if layout == "Bipartite":
                top = nx.bipartite.sets(G)[0]
                pos = nx.bipartite_layout(G,top)
            elif layout == "Circular":
                pos = nx.circular_layout(G)
            elif layout == "Fruchterman":
                pos = nx.fruchterman_reingold_layout(G)
            elif layout == "Kawai":
                pos = nx.kamada_kawai_layout(G)
            elif layout == "Planar":
                pos = nx.planar_layout(G)   
            elif layout == "Random":
                pos = nx.random_layout(G)
            elif layout == "Shell":
                pos = nx.shell_layout(G)
            elif layout =="Spectral":
                pos = nx.spectral_layout(G)
            elif layout =="Spiral":
                pos = nx.spiral_layout(G)
            elif layout=="Spring":
                pos = nx.spring_layout(G)
            else: # default
                pos = nx.planar_layout(G)
                
        else:
            # if node position are given
            pos = nx.get_node_attributes(G,'pos')
        
        
        # edges
        totalWeight=G.size(weight='weight')
        for (node1,node2,data) in G.edges(data=True):
            weight=data['weight']
            if totalWeight>0:
                width = weight*totalNodes/totalWeight*5
            else:
                width=1
            
            if arrThreshold is not None:
                if weight<arrThreshold[0]:
                    color='green'
                elif weight<arrThreshold[1]:
                    color='yellow'
                else:
                    color='red'
            else:
                color='black'
            if G.has_edge(node2,node1): # if two ways: curve
                IFN.drawCurve(pos,node1,node2,weight,width,color)
            else: # if one way: straight line
                IFN.drawArrow(pos,node1,node2,weight,width,color)
            
            # draw straight only - original
#            weighted_edges = [(node1,node2) for (node1,node2,edge_attr) in G.edges(data=True) if edge_attr['weight']==weight]
#            nx.draw_networkx_edges(G,pos, edgelist=weighted_edges, width=width,edge_color='black')        
        # nodes
        nodes=nx.draw_networkx_nodes(G, pos, node_color ='w')#, node_size=200)
        nodes.set_edgecolor('black')
        
        # labels
        nx.draw_networkx_labels(G, pos, font_size=14, font_family='sans-serif')
#        edge_labels=dict([((u,v,),round(d['weight'],2))for u,v,d in G.edges(data=True)])
#        nx.draw_networkx_edge_labels(G,pos,edge_labels=edge_labels,font_size=6, font_family='sans-serif')

        
        if routes is not None:
            edges = []
            for r in routes:
                route_edges = [(r[n],r[n+1]) for n in range(len(r)-1)]
                G.add_nodes_from(r)
                G.add_edges_from(route_edges)
                edges.append(route_edges)
            for ctr, edgelist in enumerate(edges):
                nx.draw_networkx_edges(G,pos=pos,edgelist=edgelist,edge_color = 'r',width=1.2)
          
        plt.axis('off')
        plt.title(str(self.name) + " ($\kappa$="+str(int(round(self.totalFlow(),2)))+")",{'fontsize': 26, 'fontweight' : 15})
        plt.show()
        return G
    
    @staticmethod
    def drawCurve(self,pos,node1,node2,weight,width,color):
        ax = plt.gca()
        ax.annotate("", xy=pos[node2], xycoords='data',
                        xytext=pos[node1], textcoords='data',
                        arrowprops=dict(width=0.01,shrink=0.1,
                                        color=color, linewidth=width,
                                        shrinkA=5, shrinkB=5,
                                        patchA=None, patchB=None,
                                        connectionstyle="arc3,rad=-0.25",
                                        ),
                        )
        shift=(-0.1,0)
        plt.annotate(text=weight,xy=(0,0), xytext=(0.6*np.array(pos[node2])+0.4*np.array(pos[node1]))+shift,color='red',size=16,textcoords='data')
        
    @staticmethod
    def drawArrow(pos,node1,node2,weight,width,color):
        ax = plt.gca()
        ax.annotate("", xy=pos[node2], xycoords='data',
                        xytext=pos[node1], textcoords='data',
                        arrowprops=dict(width=0.01,shrink=0.045,
                                        color=color, linewidth=width,
                                        shrinkA=5, shrinkB=5,
                                        patchA=None, patchB=None,
                                        connectionstyle="arc3,rad=-0.001",
                                        ),
                        )
        shift=(-0.08,-0.025)
        plt.annotate(text=weight,xy=(0,0), xytext=(0.56*np.array(pos[node2])+0.44*np.array(pos[node1]))+shift,color='red',size=16,textcoords='data')
#        plt.axis('off')
#        plt.show()


    @staticmethod
    def to_adjacency_matrix(matrix):
        """
        Convert non-negative matrix to (0, 1) adjacency matrix.
        
        Parameters:
        matrix (list of list of int/float): The input matrix.

        Returns:
        list of list of int: The adjacency matrix.
        """
        A = np.array(matrix) > 0
        return A.astype(int).tolist()

    @staticmethod
    def capacity2adj(C):
        """
        Convert capacity matrix to adjacency matrix.
        
        Parameters:
        C (list of list of int/float): The capacity matrix.

        Returns:
        list of list of int: The adjacency matrix.
        """
        return (np.array(C) > 0).astype(int).tolist()

    @staticmethod
    def capacity2_col_stochastic(C):
        """
        Convert capacity matrix to column stochastic matrix.
        
        Parameters:
        C (list of list of int/float): The capacity matrix.

        Returns:
        list of list of float: The column stochastic matrix.
        """
        C = np.array(C)
        if C.shape[0] != C.shape[1]:
            return "Capacity Matrix must be a square matrix."
        else:
            denom = np.sum(C, axis=0, keepdims=True)
            return (C / denom).tolist()

    @staticmethod
    def capacity2_row_stochastic(C):
        """
        Convert capacity matrix to row stochastic matrix.
        
        Parameters:
        C (list of list of int/float): The capacity matrix.

        Returns:
        list of list of float: The row stochastic matrix.
        """
        C = np.array(C)
        if C.shape[0] != C.shape[1]:
            return "Capacity Matrix must be a square matrix."
        else:
            denom = np.sum(C, axis=1, keepdims=True)
            return (C / denom).tolist()

    @staticmethod
    def capacity2stochastic(C, alpha=1, beta=0.00001):
        """
        Convert capacity matrix to a stochastic matrix with parameters alpha and beta.
        
        Parameters:
        C (list of list of int/float): The capacity matrix.
        alpha (float): The alpha parameter.
        beta (float): The beta parameter.

        Returns:
        list of list of float: The stochastic matrix.
        """
        C = np.array(C)
        if C.shape[0] != C.shape[1]:
            return "Capacity Matrix must be a square matrix."
        else:
            C_alpha = np.power(C, alpha)
            C_beta = np.exp(beta * C)
            C_transformed = C_alpha * C_beta
            denom = np.sum(C_transformed, axis=1, keepdims=True)
            return (C_transformed / denom).tolist()

    @staticmethod
    def to_equal_outflow(C):
        """
        Return ideal flow matrix with equal outflow from capacity matrix.
        
        Parameters:
        C (list of list of int/float): The capacity matrix.

        Returns:
        list of list of int/float: The ideal flow matrix.
        """
        A = IFN.to_adjacency_matrix(C)
        S = IFN.capacity2_row_stochastic(A)
        F = IFN.stochastic2_ideal_flow(S)
        scaling = IFN.global_scaling(F, 'int')
        return IFN.equivalent_ifn(F, scaling)

    @staticmethod
    def to_equal_inflow(C):
        """
        Return ideal flow matrix with equal inflow from capacity matrix.
        
        Parameters:
        C (list of list of int/float): The capacity matrix.

        Returns:
        list of list of int/float: The ideal flow matrix.
        """
        A = np.transpose(IFN.to_adjacency_matrix(C))
        S = IFN.capacity2_row_stochastic(A)
        F = np.transpose(IFN.stochastic2_ideal_flow(S))
        scaling = IFN.global_scaling(F, 'int')
        return IFN.equivalent_ifn(F, scaling)

    @staticmethod
    def capacity2_balance_inflow_outflow(C, lambda_=0.5):
        """
        Return ideal flow matrix balancing inflow and outflow from capacity matrix.
        
        Parameters:
        C (list of list of int/float): The capacity matrix.
        lambda_ (float): The lambda parameter to balance inflow and outflow.

        Returns:
        list of list of int/float: The balanced ideal flow matrix.
        """
        F_in = IFN.to_equal_inflow(C)
        F_out = IFN.to_equal_outflow(C)
        F = (1 - lambda_) * np.array(F_in) + lambda_ * np.array(F_out)
        return F.tolist()

    @staticmethod
    def rand_capacity(N):
        """
        Generate random irreducible positive weighted adjacency matrix.
        
        Parameters:
        N (int): Number of nodes.

        Returns:
        list of list of int: The random capacity matrix.
        """
        n = int(N)
        m = n + int(np.random.rand() * 2 * n)
        A = IFN.rand_irreducible(n, m)
        R = np.random.randint(1, 11, (n, n))
        return (R * A).tolist()

    @staticmethod
    def random_irreducible_stochastic(N):
        """
        Generate random irreducible stochastic matrix.
        
        Parameters:
        N (int): Size of the matrix.

        Returns:
        list of list of float: The random stochastic matrix.
        """
        C = IFN.rand_capacity(N)
        Fstar = IFN.premier_ifn(C)
        denom = np.sum(Fstar, axis=1, keepdims=True)
        return (Fstar / denom).tolist()

    @staticmethod
    def matrix_random_ideal_flow(N, kappa=1):
        """
        Generate random irreducible premagic positive weighted adjacency matrix.
        
        Parameters:
        N (int): Size of the matrix.
        kappa (float): The kappa parameter.

        Returns:
        list of list of float: The random ideal flow matrix.
        """
        C = IFN.rand_capacity(N)
        S = IFN.capacity2_row_stochastic(C)
        return IFN.stochastic2_ideal_flow(S, kappa)

    @staticmethod
    def arr_sequence2_markov(arr):
        """
        Return capacity matrix (Markov matrix of frequency) from array of sequence.
        
        Parameters:
        arr (list of int): The input sequence.

        Returns:
        tuple: The Markov matrix and unique elements array.
        """
        unique, indices = np.unique(arr, return_inverse=True)
        M = np.zeros((len(unique), len(unique)))
        for (i, j) in zip(indices, indices[1:]):
            M[i, j] += 1
        return M.tolist(), unique.tolist()

    @staticmethod
    def random_walk(m_capacity, arr_name, prev_index):
        """
        Perform a random walk based on Markov capacity matrix.
        
        Parameters:
        m_capacity (list of list of int/float): The capacity matrix.
        arr_name (list of str): The names of each row or column.
        prev_index (int): Index of the previous iteration.

        Returns:
        tuple: The next node name and index.
        """
        row = prev_index
        cnt = m_capacity[row]
        prob = np.cumsum(cnt / np.sum(cnt))
        r = np.random.rand()
        index = np.searchsorted(prob, r)
        return arr_name[index], index

    
    def random_walk1(self,start_node, length=1):
        """
        Return list of nodes passed by random walk from startNode.
        
        Parameters:
        start_node (str): The starting node.
        length (int): The length of the walk.

        Returns:
        list of str: The nodes passed by the random walk.
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


    def random_walk_cycle(self, start_end_node):
        """
        Return list of nodes passed by random cycle from startNode.
        
        Parameters:
        start_end_node (str): The start and end node.

        Returns:
        list of str: The nodes passed by the random cycle.
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
    def weighted_random_choice(list_nodes, probs):
        """
        Select a random choice from the list based on given probabilities.
        
        Parameters:
        list_nodes (list of str): The list of nodes.
        probs (list of float): The list of probabilities.

        Returns:
        str: The selected node.
        """
        r = np.random.rand()
        cumulative = np.cumsum(probs)
        for i, prob in enumerate(cumulative):
            if r <= prob:
                return list_nodes[i]

    @staticmethod
    def matrix_round2_integer(matrix):
        """
        Round each element of the matrix to the nearest integer.
        
        Parameters:
        matrix (list of list of float): The input matrix.

        Returns:
        list of list of int: The rounded matrix.
        """
        return np.round(matrix).tolist()

    @staticmethod
    def is_list_larger_than(list_, num=1):
        """
        Check if all elements in the list are larger than a given number.
        
        Parameters:
        list_ (list of int/float): The input list.
        num (int/float): The number to compare against.

        Returns:
        bool: True if all elements are larger, otherwise False.
        """
        return all(x > num for x in list_)

    @staticmethod
    def flows_in_cycle(F, cycle):
        """
        Return list of flows in a cycle.
        
        Parameters:
        F (list of list of int/float): The flow matrix.
        cycle (str): The cycle string.

        Returns:
        list of int/float: The list of flows in the cycle.
        """
        list_flow = []
        for i in range(len(cycle)):
            row, col = IFN.string_link2_coord(cycle[i] + cycle[(i + 1) % len(cycle)])
            list_flow.append(F[row][col])
        return list_flow

    @staticmethod
    def change_flow_in_cycle(F, cycle, change=1):
        """
        Add (or subtract) flow matrix based on cycle.
        
        Parameters:
        F (list of list of int/float): The flow matrix.
        cycle (str): The cycle string.
        change (int/float): The amount to change the flow.

        Returns:
        list of list of int/float: The updated flow matrix.
        """
        for i in range(len(cycle)):
            row, col = IFN.string_link2_coord(cycle[i] + cycle[(i + 1) % len(cycle)])
            F[row][col] += change
        return F

    @staticmethod
    def matrix_apply_cycle(flow_matrix, cycle, flow=1):
        """
        Return updated flow matrix after applying flow unit along the given cycle.
        
        Parameters:
        flow_matrix (list of list of int/float): The flow matrix.
        cycle (str): The cycle string.
        flow (int/float): The flow to apply.

        Returns:
        list of list of int/float: The updated flow matrix.
        """
        new_flow_matrix = np.array(flow_matrix, copy=True)
        for j in range(len(cycle)):
            from_node = ord(cycle[j]) - 97
            to_node = ord(cycle[(j + 1) % len(cycle)]) - 97
            new_flow_matrix[from_node, to_node] += flow
        return new_flow_matrix.tolist()

    @staticmethod
    def string_link2_coord(str_link):
        """
        Convert link string to coordinate.
        
        Parameters:
        str_link (str): The link string.

        Returns:
        list of int: The coordinates.
        """
        return [ord(char) - 97 for char in str_link]

    @staticmethod
    def premier_ifn(C):
        """
        Return min integer IFN regardless of stochastic matrix.
        
        Parameters:
        C (list of list of int/float): The capacity matrix.

        Returns:
        list of list of int/float: The premier IFN.
        """
        size = np.shape(C)
        mR, mC = size[0], size[1]
        list_cycles = IFN.find_all_cycles_in_matrix(C)
        F = np.zeros((mR, mC))
        for cycle in list_cycles:
            F = IFN.change_flow_in_cycle(F, cycle, +1)
        return F.tolist()

    @staticmethod
    def abs_diff_capacity_flow(C, F, w1=1, w2=1):
        """
        Return scalar cost of total change between capacity and flow matrix.
        
        Parameters:
        C (list of list of int/float): The capacity matrix.
        F (list of list of int/float): The flow matrix.
        w1 (int/float): Weight for negative difference.
        w2 (int/float): Weight for positive difference.

        Returns:
        float: The total cost.
        """
        C = np.array(C)
        F = np.array(F)
        diff = C - F
        cost = np.sum(w2 * diff[diff > 0]) - np.sum(w1 * diff[diff < 0])
        return cost

    @staticmethod
    def is_edge_in_cycle(i, j, cycle):
        """
        Check if an edge is in a cycle.
        
        Parameters:
        i (int): The row index.
        j (int): The column index.
        cycle (str): The cycle string.

        Returns:
        bool: True if the edge is in the cycle, otherwise False.
        """
        n = len(cycle)
        for k in range(n):
            from_node = ord(cycle[k]) - 97
            to_node = ord(cycle[(k + 1) % n]) - 97
            if from_node == i and to_node == j:
                return True
        return False

    @staticmethod
    def adj2stochastic(A):
        """
        Convert adjacency matrix to stochastic matrix.
        
        Parameters:
        A (list of list of int): The adjacency matrix.

        Returns:
        list of list of float: The stochastic matrix.
        """
        v = np.sum(A, axis=1)
        D = np.diag(v)
        return np.dot(np.linalg.inv(D), A).tolist()

    @staticmethod
    def ideal_flow2stochastic(F):
        """
        Convert ideal flow matrix to stochastic matrix.
        
        Parameters:
        F (list of list of int/float): The ideal flow matrix.

        Returns:
        list of list of float: The stochastic matrix.
        """
        s = np.sum(F, axis=1)
        return (F / s[:, np.newaxis]).tolist()

    @staticmethod
    def ideal_flow(S, pi):
        """
        Compute the ideal flow from stochastic matrix and Perron vector.
        
        Parameters:
        S (list of list of float): The stochastic matrix.
        pi (list of float): The Perron vector.

        Returns:
        list of list of float: The ideal flow matrix.
        """
        return (S * np.array(pi)[:, np.newaxis]).tolist()

    @staticmethod
    def adj2ideal_flow(A, kappa=1):
        """
        Convert adjacency matrix to ideal flow matrix.
        
        Parameters:
        A (list of list of int): The adjacency matrix.
        kappa (float): The kappa parameter.

        Returns:
        list of list of float: The ideal flow matrix.
        """
        S = IFN.adj2stochastic(A)
        pi = IFN.Markov(S, kappa)
        return IFN.ideal_flow(S, pi)

    @staticmethod
    def capacity2ideal_flow(C, kappa=1):
        """
        Convert capacity matrix to ideal flow matrix.
        
        Parameters:
        C (list of list of int/float): The capacity matrix.
        kappa (float): The kappa parameter.

        Returns:
        list of list of float: The ideal flow matrix.
        """
        S = IFN.capacity2_row_stochastic(C)
        pi = IFN.Markov(S, kappa)
        return IFN.ideal_flow(S, pi)

    
    @staticmethod
    def stochastic2_phi(S, kappa=1):
        """
        Compute Perron vector (phi) from stochastic matrix.
        
        Parameters:
        S (list of list of float): The stochastic matrix.
        kappa (float): The kappa parameter.

        Returns:
        list of float: The Perron vector.
        """
        return IFN.Markov(S, kappa)

    @staticmethod
    def stationary_markov_chain(S):
        """
        Compute the stationary distribution of a Markov chain.
        
        Parameters:
        S (list of list of float): The stochastic matrix.

        Returns:
        list of float: The stationary distribution.
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
        return v.flatten().tolist()

    @staticmethod
    def stochastic2_ideal_flow(S, kappa=1):
        """
        Convert stochastic matrix to ideal flow matrix.
        
        Parameters:
        S (list of list of float): The stochastic matrix.
        kappa (float): The kappa parameter.

        Returns:
        list of list of float: The ideal flow matrix.
        """
        S = np.array(S)
        mR, mC = S.shape
        if mR != mC:
            return "Stochastic Matrix must be a square matrix."
        if mR < 15:
            phi = IFN.stochastic2_phi(S, kappa)
            phiJt = np.dot(np.array(phi)[:, np.newaxis], np.ones((1, mR)))
            return (phiJt * S).tolist()
        else:
            pi = IFN.stationary_markov_chain(S)
            B = np.dot(np.array(pi)[:, np.newaxis], np.ones((1, mR)))
            return (B * S * kappa).tolist()

    @staticmethod
    def sum_of_row(M):
        """
        Compute the sum of each row in a matrix.
        
        Parameters:
        M (list of list of int/float): The input matrix.

        Returns:
        list of int/float: The row sums.
        """
        return np.sum(M, axis=1).tolist()

    @staticmethod
    def sum_of_col(M):
        """
        Compute the sum of each column in a matrix.
        
        Parameters:
        M (list of list of int/float): The input matrix.

        Returns:
        list of int/float: The column sums.
        """
        return np.sum(M, axis=0).tolist()

    @staticmethod
    def is_square(M):
        """
        Check if a matrix is square.
        
        Parameters:
        M (list of list of int/float): The input matrix.

        Returns:
        bool: True if the matrix is square, otherwise False.
        """
        return np.shape(M)[0] == np.shape(M)[1]

    @staticmethod
    def is_non_negative(M):
        """
        Check if all elements in a matrix are non-negative.
        
        Parameters:
        M (list of list of int/float): The input matrix.

        Returns:
        bool: True if all elements are non-negative, otherwise False.
        """
        return np.all(np.array(M) >= 0)

    @staticmethod
    def is_positive(M):
        """
        Check if all elements in a matrix are positive.
        
        Parameters:
        M (list of list of int/float): The input matrix.

        Returns:
        bool: True if all elements are positive, otherwise False.
        """
        return np.all(np.array(M) > 0)

    # @staticmethod
    # def is_premagic(M):
    #     """
    #     Check if a matrix is a premagic matrix.
        
    #     Parameters:
    #     M (list of list of int/float): The input matrix.

    #     Returns:
    #     bool: True if the matrix is premagic, otherwise False.
    #     """
    #     return IFN.is_premagic(M)

    # def is_irreducible(self, M):
    #     """
    #     Check if a matrix is irreducible.
        
    #     Parameters:
    #     M (list of list of int/float): The input matrix.

    #     Returns:
    #     bool: True if the matrix is irreducible, otherwise False.
    #     """
    #     return self.is_irreducible(M)

    # def is_ideal_flow(self, M):
    #     """
    #     Check if a matrix is an ideal flow matrix.
        
    #     Parameters:
    #     M (list of list of int/float): The input matrix.

    #     Returns:
    #     bool: True if the matrix is an ideal flow matrix, otherwise False.
    #     """
    #     return self.is_ideal_flow(M)

    @staticmethod
    def is_premier(F):
        """
        Check if a flow matrix is premier.
        
        Parameters:
        F (list of list of int/float): The flow matrix.

        Returns:
        bool: True if the matrix is premier, otherwise False.
        """
        net_signature = IFN.decompose(F)
        return IFN._is_net_signature_premier(net_signature)

    @staticmethod
    def equivalent_ifn(F, scaling, is_rounded=True):
        """
        Compute equivalent ideal flow network with given scaling.
        
        Parameters:
        F (list of list of int/float): The flow matrix.
        scaling (float): The scaling factor.
        is_rounded (bool): Whether to round the result.

        Returns:
        list of list of int/float: The equivalent ideal flow matrix.
        """
        F = np.array(F)
        if is_rounded:
            return np.round(F * scaling).tolist()
        return (F * scaling).tolist()

    @staticmethod
    def global_scaling(F, scaling_type='min', val=1):
        """
        Compute global scaling factor for a flow matrix.
        
        Parameters:
        F (list of list of int/float): The flow matrix.
        scaling_type (str): The type of scaling ('min', 'max', 'sum', 'int', 'avg', 'std', 'cov').
        val (float): The value for scaling.

        Returns:
        float: The scaling factor.
        """
        f = np.array(F).flatten()
        f = f[f > 0]
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
            common_denominator = np.lcm.reduce([frac[1] for frac in fractions])
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
    def scale_array_to_integer_ratios(arr):
        """
        Scale array to integer ratios.
        
        Parameters:
        arr (list of float): The input array.

        Returns:
        list of int: The scaled array.
        """
        fractions = [IFN.decimal_to_fraction(x) for x in arr]
        common_denominator = np.lcm.reduce([frac[1] for frac in fractions])
        return [(frac[0] * common_denominator) // frac[1] for frac in fractions]

    @staticmethod
    def min_irreducible(k):
        """
        Generate minimum irreducible matrix of size k.
        
        Parameters:
        k (int): The size of the matrix.

        Returns:
        list of list of int: The minimum irreducible matrix.
        """
        A = np.zeros((k, k))
        for r in range(k - 1):
            A[r, r + 1] = 1
        A[k - 1, 0] = 1
        return A.tolist()
    
    @staticmethod
    def add_random_ones(A, m=6):
        """
        Add random ones to a matrix.
        
        Parameters:
        A (list of list of int): The input matrix.
        m (int): Number of ones to add.

        Returns:
        list of list of int: The updated matrix.
        """
        n = len(A)
        n2 = np.sum(A)
        if m > n2:
            k = 0
            while k < m - n2:
                idx = np.random.randint(0, n * n)
                row, col = divmod(idx, n)
                if A[row][col] == 0:
                    A[row][col] = 1
                    k += 1
        return A

    @staticmethod
    def rand_permutation_eye(n=5):
        """
        Generate random permutation of identity matrix.
        
        Parameters:
        n (int): Size of the matrix.

        Returns:
        list of list of int: The permuted identity matrix.
        """
        eye = np.eye(n)
        np.random.shuffle(eye)
        return eye.tolist()

    @staticmethod
    def rnd_ifn4_capacity(num_node=5):
        """
        Generate random ideal flow network for a given number of nodes.
        
        Parameters:
        num_node (int): Number of nodes.

        Returns:
        list of list of int/float: The ideal flow network.
        """
        max_capacity = 9
        C = IFN.rnd_capacity(num_node, max_capacity)
        F = IFN.capacity2_ideal_flow(C)
        scaling = IFN.global_scaling(F, 'int')
        return IFN.equivalent_ifn(F, scaling)

    @staticmethod
    def rnd_ifn(num_nodes=5, total_flow=0):
        """
        Generate random ideal flow network with given total flow.
        
        Parameters:
        num_nodes (int): Number of nodes.
        total_flow (int): Total flow.

        Returns:
        list of list of int/float: The ideal flow network.
        """
        if total_flow <= 0:
            total_flow = np.random.randint(10, 100)
        signature = IFN.find_ifn_signature(num_nodes, total_flow)
        return IFN.compose(signature)

    @staticmethod
    def kappa(F):
        """
        Compute kappa value of a flow matrix.
        
        Parameters:
        F (list of list of int/float): The flow matrix.

        Returns:
        float: The kappa value.
        """
        return np.sum(F)

    @staticmethod
    def min_flow(self, F):
        """
        Compute minimum flow value of a flow matrix.
        
        Parameters:
        F (list of list of int/float): The flow matrix.

        Returns:
        float: The minimum flow value.
        """
        f = np.array(F).flatten()
        f = f[f > 0]
        return np.min(f)

    @staticmethod
    def max_flow(self, F):
        """
        Compute maximum flow value of a flow matrix.
        
        Parameters:
        F (list of list of int/float): The flow matrix.

        Returns:
        float: The maximum flow value.
        """
        f = np.array(F).flatten()
        f = f[f > 0]
        return np.max(f)

    @staticmethod
    def avg_flow(self, F):
        """
        Compute average flow value of a flow matrix.
        
        Parameters:
        F (list of list of int/float): The flow matrix.

        Returns:
        float: The average flow value.
        """
        f = np.array(F).flatten()
        f = f[f > 0]
        return np.mean(f)

    @staticmethod
    def std_flow(self, F):
        """
        Compute standard deviation of flow values in a flow matrix.
        
        Parameters:
        F (list of list of int/float): The flow matrix.

        Returns:
        float: The standard deviation of flow values.
        """
        f = np.array(F).flatten()
        f = f[f > 0]
        return np.std(f)

    @staticmethod
    def coef_var_flow(self, F):
        """
        Compute coefficient of variation of flow values in a flow matrix.
        
        Parameters:
        F (list of list of int/float): The flow matrix.

        Returns:
        float: The coefficient of variation.
        """
        f = np.array(F).flatten()
        f = f[f > 0]
        return np.std(f) / np.mean(f)

    @staticmethod
    def capacity2_congestion(C, kappa, capacity_multiplier):
        """
        Compute congestion matrix from capacity matrix and kappa.
        
        Parameters:
        C (list of list of int/float): The capacity matrix.
        kappa (float): The kappa parameter.
        capacity_multiplier (float): The capacity multiplier.

        Returns:
        list of list of float: The congestion matrix.
        """
        S = IFN.capacity2_row_stochastic(C)
        F = IFN.stochastic2_ideal_flow(S, kappa)
        return (np.array(F) / (np.array(C) * capacity_multiplier)).tolist()

    @staticmethod
    def stochastic2_probability(S):
        """
        Compute probability matrix from stochastic matrix.
        
        Parameters:
        S (list of list of float): The stochastic matrix.

        Returns:
        list of float: The probability matrix.
        """
        phi = IFN.stochastic2_phi(S, 1)
        phiJt = np.dot(np.array(phi)[:, np.newaxis], np.ones((1, len(S))))
        return (S * phiJt).tolist()

    @staticmethod
    def stochastic2_network_entropy(S):
        """
        Compute network entropy from stochastic matrix.
        
        Parameters:
        S (list of list of float): The stochastic matrix.

        Returns:
        float: The network entropy.
        """
        s = np.array(S).flatten()
        s = s[s > 0]
        return -np.sum(s * np.log(s))

    @staticmethod
    def stochastic2_entropy_ratio(S):
        """
        Compute entropy ratio from stochastic matrix.
        
        Parameters:
        S (list of list of float): The stochastic matrix.

        Returns:
        float: The entropy ratio.
        """
        h1 = IFN.network_entropy(S)
        A = np.array(S) > 0
        T = IFN.adj2stochastic(A)
        h0 = IFN.network_entropy(T)
        return h1 / h0

    @staticmethod
    def max_network_entropy(P):
        """
        Compute maximum network entropy for a given probability matrix.
        
        Parameters:
        P (list of list of float): The probability matrix.

        Returns:
        tuple: The entropy, entropy ratio, and maximum entropy.
        """
        P = np.array(P)
        arr_prob = P[P > 0]
        n = len(arr_prob)
        p_uniform = 1 / n
        entropy = -np.sum(arr_prob * np.log(arr_prob))
        max_ent = -np.sum([p_uniform * np.log(p_uniform) for _ in range(n)])
        return entropy, entropy / max_ent, max_ent

    @staticmethod
    def network_entropy(F):
        """
        Compute network entropy for a given flow matrix.
        
        Parameters:
        F (list of list of int/float): The flow matrix.

        Returns:
        float: The network entropy.
        """
        total_flow = IFN.kappa(F)
        entropy = 0
        for row in F:
            for val in row:
                if val > 0:
                    p = val / total_flow
                    entropy -= p * np.log(p)
        return entropy

    @staticmethod
    def average_node_entropy(S):
        """
        Compute average node entropy from stochastic matrix.
        
        Parameters:
        S (list of list of float): The stochastic matrix.

        Returns:
        float: The average node entropy.
        """
        S = np.array(S)
        positive_list = S[S > 0]
        if len(positive_list) == 0:
            return None
        total = np.sum(positive_list)
        probabilities = positive_list / total
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy / S.shape[0]

    @staticmethod
    def avg_node_entropy(F):
        """
        Compute average node entropy for a given flow matrix.
        
        Parameters:
        F (list of list of int/float): The flow matrix.

        Returns:
        float: The average node entropy.
        """
        F = np.array(F)
        node_entropies = []
        for row in F:
            row_sum = np.sum(row)
            entropy = 0
            if row_sum > 0:
                probabilities = row / row_sum
                entropy = -np.sum(probabilities * np.log(probabilities))
            node_entropies.append(entropy)
        return np.mean(node_entropies)

    @staticmethod
    def avg_node_entropy_ratio(F):
        """
        Compute average node entropy ratio for a given flow matrix.
        
        Parameters:
        F (list of list of int/float): The flow matrix.

        Returns:
        float: The average node entropy ratio.
        """
        actual_entropy = IFN.avg_node_entropy(F)
        total_nodes = F.shape[0]
        max_entropy = 0
        for row in F:
            active_connections = np.sum(row > 0)
            if active_connections > 0:
                max_entropy += np.log(active_connections)
        return actual_entropy / (max_entropy / total_nodes)

    @staticmethod
    def network_entropy_ratio(F):
        """
        Compute network entropy ratio for a given flow matrix.
        
        Parameters:
        F (list of list of int/float): The flow matrix.

        Returns:
        float: The network entropy ratio.
        """
        total_flows = np.sum(np.array(F) > 0)
        actual_entropy = IFN.network_entropy(F)
        max_network_entropy = np.log(total_flows)
        return actual_entropy / max_network_entropy

    @staticmethod
    def rand_irreducible(k=5, m=8):
        """
        Generate random irreducible matrix of size k with m additional links.
        
        Parameters:
        k (int): Size of the matrix.
        m (int): Number of additional links.

        Returns:
        list of list of int: The random irreducible matrix.
        """
        A = IFN.min_irreducible(k)
        A = IFN.add_random_ones(A, m)
        P = IFN.rand_permutation_eye(k)
        A2 = np.dot(np.dot(P, A), P.T)
        return (A2 > 0).astype(int).tolist()

    @staticmethod
    def rnd_capacity(num_node=5, max_capacity=9):
        """
        Generate random capacity matrix for a given number of nodes.
        
        Parameters:
        num_node (int): Number of nodes.
        max_capacity (int): Maximum capacity value.

        Returns:
        list of list of int: The random capacity matrix.
        """
        num_link = np.random.randint(1, num_node // 2 + 1) * num_node + 1
        C = IFN.rand_irreducible(num_node, num_link)
        for i in range(num_node):
            C[i][i] = 0
        for i in range(num_node):
            for j in range(num_node):
                if C[i][j] > 0:
                    C[i][j] = np.random.randint(1, max_capacity + 1)
        return C

    @staticmethod
    def is_equal_signature(signature1, signature2):
        """
        Check if two signatures are equal.
        
        Parameters:
        signature1 (str): The first signature.
        signature2 (str): The second signature.

        Returns:
        bool: True if the signatures are equal, otherwise False.
        """
        cycle_dict1 = IFN._parse_terms_to_dict(signature1)
        canon_cycle_dict1 = IFN.canonize_cycle_dict(cycle_dict1)
        cycle_dict2 = IFN._parse_terms_to_dict(signature2)
        canon_cycle_dict2 = IFN.canonize_cycle_dict(cycle_dict2)
        return canon_cycle_dict1 == canon_cycle_dict2

    @staticmethod
    def is_equal_sets(a, b):
        """
        Check if two sets are equal.
        
        Parameters:
        a (set): The first set.
        b (set): The second set.

        Returns:
        bool: True if the sets are equal, otherwise False.
        """
        return a == b

    @staticmethod
    def extract_first_k_terms(net_signature, k):
        """
        Extract the first k terms from a network signature.
        
        Parameters:
        net_signature (str): The network signature.
        k (int): Number of terms to extract.

        Returns:
        str: The extracted terms.
        """
        parts = net_signature.split('+')
        return ' + '.join(parts[:k])

    @staticmethod
    def extract_last_k_terms(net_signature, k):
        """
        Extract the last k terms from a network signature.
        
        Parameters:
        net_signature (str): The network signature.
        k (int): Number of terms to extract.

        Returns:
        str: The extracted terms.
        """
        parts = net_signature.split('+')
        return ' + '.join(parts[-k:])

    @staticmethod
    def generate_random_terms(net_signature, k, is_premier=False):
        """
        Generate k random terms from a network signature.
        
        Parameters:
        net_signature (str): The network signature.
        k (int): Number of terms to generate.
        is_premier (bool): Whether the terms are premier.

        Returns:
        str: The generated terms.
        """
        cycle_dict = IFN._parse_terms_to_dict(net_signature)
        terms = list(cycle_dict.keys())
        np.random.shuffle(terms)
        selected_terms = terms[:k]
        if is_premier:
            return ' + '.join(selected_terms)
        return ' + '.join(f"{np.random.randint(1, 11)}{term}" for term in selected_terms)

    @staticmethod
    def premier_signature(C):
        """
        Compute premier signature for a given capacity matrix.
        
        Parameters:
        C (list of list of int/float): The capacity matrix.

        Returns:
        str: The premier signature.
        """
        cycles = IFN.find_all_cycles_in_matrix(C)
        return ' + '.join(cycles)

    @staticmethod
    def canonize_net_signature(net_signature):
        """
        Canonize a network signature.
        
        Parameters:
        net_signature (str): The network signature.

        Returns:
        str: The canonized signature.
        """
        node_mapping = IFN.create_node_mapping(net_signature)
        net_signature = IFN.relabel_net_signature(net_signature, node_mapping)
        cycle_dict = IFN._parse_terms_to_dict(net_signature)
        canon_cycle_dict = IFN.canonize_cycle_dict(cycle_dict)
        return IFN.cycle_dict2_signature(canon_cycle_dict)

    @staticmethod
    def create_node_mapping(net_signature):
        """
        Create node mapping for a network signature.
        
        Parameters:
        net_signature (str): The network signature.

        Returns:
        dict: The node mapping.
        """
        unique_nodes = IFN._identify_unique_nodes(net_signature)
        node_mapping = {node: chr(97 + i) for i, node in enumerate(unique_nodes)}
        return node_mapping

    @staticmethod
    def relabel_net_signature(net_signature, node_mapping):
        """
        Relabel a network signature using a node mapping.
        
        Parameters:
        net_signature (str): The network signature.
        node_mapping (dict): The node mapping.

        Returns:
        str: The relabeled signature.
        """
        return ''.join(node_mapping.get(char, char) for char in net_signature)

    @staticmethod
    def reverse_relabel_net_signature(relabeled_signature, node_mapping):
        """
        Reverse relabel a network signature using a node mapping.
        
        Parameters:
        relabeled_signature (str): The relabeled signature.
        node_mapping (dict): The node mapping.

        Returns:
        str: The original signature.
        """
        inverse_node_mapping = {v: k for k, v in node_mapping.items()}
        return ''.join(inverse_node_mapping.get(char, char) for char in relabeled_signature)

    @staticmethod
    def cycle_dict2_signature(cycle_dict):
        """
        Convert cycle dictionary to network signature.
        
        Parameters:
        cycle_dict (dict): The cycle dictionary.

        Returns:
        str: The network signature.
        """
        terms = []
        for cycle, alpha in cycle_dict.items():
            if alpha == 1:
                terms.append(cycle)
            elif alpha == -1:
                terms.append(f"(-{cycle})")
            else:
                terms.append(f"({alpha}){cycle}")
        return ' + '.join(terms)

    @staticmethod
    def is_valid_signature(signature):
        """
        Check if a network signature is valid.
        
        Parameters:
        signature (str): The network signature.

        Returns:
        bool: True if the signature is valid, otherwise False.
        """
        cycle_dict = IFN._parse_terms_to_dict(signature)
        canon_cycle_dict = IFN.canonize_cycle_dict(cycle_dict)
        return all(IFN.is_cycle_canonical(cycle) for cycle in canon_cycle_dict)

    @staticmethod
    def is_cycle_canonical(cycle):
        """
        Check if a cycle is canonical.
        
        Parameters:
        cycle (str): The cycle string.

        Returns:
        bool: True if the cycle is canonical, otherwise False.
        """
        return all(cycle[i] <= cycle[i + 1] for i in range(len(cycle) - 1))

    @staticmethod
    def is_irreducible_signature(signature):
        """
        Check if a network signature is irreducible.
        
        Parameters:
        signature (str): The network signature.

        Returns:
        bool: True if the signature is irreducible, otherwise False.
        """
        cycle_dict = IFN._parse_terms_to_dict(signature)
        canon_cycle_dict = IFN.canonize_cycle_dict(cycle_dict)
        cycles = list(canon_cycle_dict.keys())
        if len(cycles) <= 1:
            return True
        return all(
            any(IFN.has_pivot(cycle, other_cycle) for j, other_cycle in enumerate(cycles) if i != j)
            for i, cycle in enumerate(cycles)
        )

    @staticmethod
    def has_pivot(cycle1, cycle2):
        """
        Check if two cycles have a pivot.
        
        Parameters:
        cycle1 (str): The first cycle.
        cycle2 (str): The second cycle.

        Returns:
        bool: True if there is a pivot, otherwise False.
        """
        nodes1 = set(cycle1)
        nodes2 = set(cycle2)
        return bool(nodes1 & nodes2)

    @staticmethod
    def find_pivots(signature):
        """
        Find pivots in a network signature.
        
        Parameters:
        signature (str): The network signature.

        Returns:
        list of dict: The list of pivots.
        """
        cycle_dict = IFN._parse_terms_to_dict(signature)
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
    def find_pivot_type(cycle1, cycle2):
        """
        Determine the type of pivot between two cycles.
        
        Parameters:
        cycle1 (str): The first cycle.
        cycle2 (str): The second cycle.

        Returns:
        str: The type of pivot.
        """
        common_nodes = set(cycle1) & set(cycle2)
        if len(common_nodes) > 1:
            return 'path: ' + ''.join(common_nodes)
        elif len(common_nodes) == 1:
            return 'node: ' + common_nodes.pop()
        return 'no pivot'

    @staticmethod
    def canonize_cycle_dict(cycle_dict):
        """
        Canonize a cycle dictionary.
        
        Parameters:
        cycle_dict (dict): The cycle dictionary.

        Returns:
        dict: The canonized cycle dictionary.
        """
        canon_cycle_dict = {}
        for cycle, alpha in cycle_dict.items():
            canon_cycle = IFN.canonize(list(cycle))
            if canon_cycle in canon_cycle_dict:
                canon_cycle_dict[canon_cycle] += alpha
            else:
                canon_cycle_dict[canon_cycle] = alpha
        return canon_cycle_dict

    @staticmethod
    def canonize(cycle):
        """
        Canonize a cycle.
        
        Parameters:
        cycle (list of str): The cycle.

        Returns:
        list of str: The canonized cycle.
        """
        min_idx = min(range(len(cycle)), key=lambda i: cycle[i:] + cycle[:i])
        rotated = cycle[min_idx:] + cycle[:min_idx]
        reversed_cycle = rotated[::-1]
        return min(rotated, reversed_cycle)

    @staticmethod
    def link_cycle_matrix(F):
        """
        Generate link-cycle matrix from flow matrix.
        
        Parameters:
        F (list of list of int/float): The flow matrix.

        Returns:
        dict: The link-cycle matrix.
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
        return {'H': H.tolist(), 'y': y, 'cycles': cycles, 'links': links}

    @staticmethod
    def find_all_cycles_in_matrix(matrix):
        """
        Find all cycles in a given matrix.
        
        Parameters:
        matrix (list of list of int/float): The input matrix.

        Returns:
        list of str: The list of cycles.
        """
        n = len(matrix)
        adj_list = [[] for _ in range(n)]
        for i, row in enumerate(matrix):
            for j, cell in enumerate(row):
                if cell != 0:
                    adj_list[i].append(j)
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
                    cycles.add(canonical([chr(97 + node) for node in stack]))
                elif not visited[w]:
                    dfs(w, start, visited, stack)
            stack.pop()
            visited[v] = False
        for i in range(n):
            dfs(i, i, [False] * n, [])
        return list(cycles)

    @staticmethod
    def find_all_cycles_in_adj_list(adjL):
        """
        Find all cycles in a given adjacency list.
        
        Parameters:
        adjL (dict): The adjacency list.

        Returns:
        list of str: The list of cycles.
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
                    cycles.add(canonical([nodes[node] for node in stack]))
                elif not visited[w]:
                    dfs(w, start, visited, stack)
            stack.pop()
            visited[v] = False
        for i in range(len(nodes)):
            if adj_list[i]:
                dfs(i, i, [False] * len(nodes), [])
        return list(cycles)

    @staticmethod
    def find_all_walks_in_matrix(matrix):
        """
        Find all walks in a given matrix.
        
        Parameters:
        matrix (list of list of int/float): The input matrix.

        Returns:
        list of str: The list of walks.
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
    def compose(signature):
        """
        Compose flow matrix from network signature.

        Parameters:
        signature (str): The network signature.

        Returns:
        list of list of int/float: The composed flow matrix.
        """
        terms = IFN._parse_terms_to_dict(signature)
        num_nodes = IFN.signature2_num_nodes(signature)
        F = np.zeros((num_nodes, num_nodes))
        for cycle, coef in terms.items():
            cycle_nodes = IFN._parse_cycle(cycle)
            IFN._assign_cycle(F, cycle_nodes, coef)
        return F.tolist()

    @staticmethod
    def decompose(F):
        """
        Decompose flow matrix into network signature.

        Parameters:
        F (list of list of int/float): The flow matrix.

        Returns:
        str: The decomposed network signature.
        """
        adj_list = IFN.flow_matrix_to_adj_list(F)
        cycles = IFN.find_all_cycles_in_adj_list(adj_list)
        cycle_dict = {}
        for cycle in cycles:
            cycle_value = min(IFN.flows_in_cycle(F, cycle))
            if cycle_value > 0:
                cycle_dict[cycle] = cycle_value
                F = IFN.change_flow_in_cycle(F, cycle, -cycle_value)
        return IFN.cycle_dict2_signature(cycle_dict)

    @staticmethod
    def flow_matrix_to_adj_list(self, F):
        """
        Convert flow matrix to adjacency list.

        Parameters:
        F (list of list of int/float): The flow matrix.

        Returns:
        dict: The adjacency list.
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
    def decimal_to_fraction(x, max_denominator=10000):
        """
        Convert a decimal number to a fraction.

        Parameters:
        x (float): The decimal number.
        max_denominator (int): The maximum denominator for the fraction.

        Returns:
        tuple: The numerator and denominator of the fraction.
        """
        return (np.round(x * max_denominator), max_denominator)

    @staticmethod
    def _parse_terms_to_dict(net_signature):
        """
        Parse network signature into a dictionary of terms.

        Parameters:
        net_signature (str): The network signature.

        Returns:
        dict: The dictionary of terms.
        """
        terms = net_signature.split('+')
        term_dict = {}
        for term in terms:
            match = re.match(r'^(\d*\.?\d*)?([a-z]+)$', term.strip())
            if match:
                count_str = match[1] or '1'
                count = float(count_str) if '.' in count_str else int(count_str)
                cycle = match[2]
                term_dict[cycle] = term_dict.get(cycle, 0) + count
        return term_dict


    @staticmethod
    def _find_common_nodes(cycle1, cycle2):
        """
        Find common nodes between two cycles.

        Parameters:
        cycle1 (str): The first cycle.
        cycle2 (str): The second cycle.

        Returns:
        list of str: The common nodes.
        """
        return list(set(cycle1) & set(cycle2))

    @staticmethod
    def _canonical_cycle(cycle, node_mapping):
        """
        Canonize a cycle with a given node mapping.

        Parameters:
        cycle (list of str): The cycle.
        node_mapping (dict): The node mapping.

        Returns:
        list of str: The canonized cycle.
        """
        mapped_cycle = [node_mapping[node] for node in cycle]
        min_idx = min(range(len(mapped_cycle)), key=lambda i: mapped_cycle[i:] + mapped_cycle[:i])
        rotated = mapped_cycle[min_idx:] + mapped_cycle[:min_idx]
        reversed_cycle = rotated[::-1]
        return min(rotated, reversed_cycle)

    @staticmethod
    def _is_net_signature_premier(net_signature):
        """
        Check if a network signature is premier.

        Parameters:
        net_signature (str): The network signature.

        Returns:
        bool: True if the signature is premier, otherwise False.
        """
        signature1 = IFN._convert_signature_coef2one(net_signature)
        F = IFN.compose(signature1)
        cycles = IFN.find_all_cycles_in_matrix(F)
        cycle_dict = IFN._parse_terms_to_dict(net_signature)
        canon_cycle_dict = IFN.canonize_cycle_dict(cycle_dict)
        cycle_set = set(cycles)
        return all(cycle_set and canon_cycle_dict[cycle] == 1 for cycle in canon_cycle_dict) and len(cycle_set) == len(canon_cycle_dict)

    @staticmethod
    def _convert_signature_coef2one(net_signature):
        """
        Convert network signature coefficients to one.

        Parameters:
        net_signature (str): The network signature.

        Returns:
        str: The converted signature.
        """
        cycle_dict = IFN._parse_terms_to_dict(net_signature)
        return ' + '.join(cycle_dict)

    @staticmethod
    def _parse_cycle(cycle):
        """
        Parse cycle string into a list of node indices.

        Parameters:
        cycle (str): The cycle string.

        Returns:
        list of int: The node indices.
        """
        return [ord(name) - 97 for name in cycle]

    @staticmethod
    def _assign_cycle(F, cycle, value):
        """
        Assign a value to the edges in a cycle within the flow matrix.

        Parameters:
        F (list of list of int/float): The flow matrix.
        cycle (list of int): The cycle as a list of node indices.
        value (int/float): The value to assign.

        Returns: F by reference
        None
        """
        for i in range(len(cycle) - 1):
            F[cycle[i]][cycle[i + 1]] += value
        F[cycle[-1]][cycle[0]] += value

    @staticmethod
    def _identify_unique_nodes(net_signature):
        """
        Identify unique nodes in a network signature.

        Parameters:
        net_signature (str): The network signature.

        Returns:
        list of str: The unique nodes.
        """
        return sorted(set(char for term in net_signature.split('+') for char in term if char.isalpha()))
    
    @staticmethod
    def signature2link_flow(cycle_signature, is_cycle=True):
        """
        Compute link flow values from cycle signature.
        
        Parameters:
        cycle_signature (str): The cycle signature.
        is_cycle (bool): Whether the signature represents a cycle.

        Returns:
        dict: The link flow values.
        """
        terms = IFN._parse_terms_to_dict(cycle_signature)
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
    def signature2_num_nodes(cycle_signature):
        """
        Compute the number of nodes in a cycle signature.
        
        Parameters:
        cycle_signature (str): The cycle signature.

        Returns:
        int: The number of nodes.
        """
        return len(set(char for term in cycle_signature.split('+') for char in term if char.isalpha()))

    @staticmethod
    def signature2_links(cycle_signature, is_cycle=True):
        """
        Compute the links in a cycle signature.
        
        Parameters:
        cycle_signature (str): The cycle signature.
        is_cycle (bool): Whether the signature represents a cycle.

        Returns:
        set: The set of links.
        """
        terms = IFN._parse_terms_to_dict(cycle_signature)
        unique_links = set()
        for cycle in terms:
            nodes = cycle
            for i in range(len(nodes)):
                link = nodes[i] + nodes[(i + 1) % len(nodes)] if is_cycle else nodes[i] + nodes[i + 1] if i + 1 < len(nodes) else None
                if link:
                    unique_links.add(link)
        return unique_links

    @staticmethod
    def signature2_num_links(cycle_signature, is_cycle=True):
        """
        Compute the number of links in a cycle signature.
        
        Parameters:
        cycle_signature (str): The cycle signature.
        is_cycle (bool): Whether the signature represents a cycle.

        Returns:
        int: The number of links.
        """
        return len(IFN.signature2_links(cycle_signature, is_cycle))

    @staticmethod
    def signature2_row_stochastic(cycle_signature, is_cycle=True):
        """
        Convert cycle signature to row stochastic adjacency list.
        
        Parameters:
        cycle_signature (str): The cycle signature.
        is_cycle (bool): Whether the signature represents a cycle.

        Returns:
        dict: The row stochastic adjacency list.
        """
        link_flows = IFN.signature2link_flow(cycle_signature, is_cycle)
        row_sums = IFN.signature2_sum_rows(cycle_signature)
        stocastic_adj_list = {}
        for link, flow in link_flows.items():
            source, target = link
            if source not in stocastic_adj_list:
                stocastic_adj_list[source] = {}
            stocastic_adj_list[source][target] = str(flow / row_sums[source])
        return stocastic_adj_list

    @staticmethod
    def signature2_column_stochastic(cycle_signature, is_cycle=True):
        """
        Convert cycle signature to column stochastic adjacency list.
        
        Parameters:
        cycle_signature (str): The cycle signature.
        is_cycle (bool): Whether the signature represents a cycle.

        Returns:
        dict: The column stochastic adjacency list.
        """
        link_flows = IFN.signature2link_flow(cycle_signature, is_cycle)
        col_sums = IFN.signature2_sum_cols(cycle_signature)
        stocastic_adj_list = {}
        for link, flow in link_flows.items():
            source, target = link
            if source not in stocastic_adj_list:
                stocastic_adj_list[source] = {}
            stocastic_adj_list[source][target] = str(flow / col_sums[target])
        return stocastic_adj_list

    @staticmethod
    def signature2_ideal_flow(cycle_signature, is_cycle=True):
        """
        Convert cycle signature to ideal flow adjacency list.
        
        Parameters:
        cycle_signature (str): The cycle signature.
        is_cycle (bool): Whether the signature represents a cycle.

        Returns:
        dict: The ideal flow adjacency list.
        """
        return IFN.signature2_adj_list(cycle_signature, is_cycle)

    @staticmethod
    def signature2_kappa(cycle_signature):
        """
        Compute the kappa value of a cycle signature.
        
        Parameters:
        cycle_signature (str): The cycle signature.

        Returns:
        int/float: The kappa value.
        """
        terms = IFN._parse_terms_to_dict(cycle_signature)
        return sum(alpha * len(cycle) for cycle, alpha in terms.items())

    @staticmethod
    def signature2_coef_flow(cycle_signature):
        """
        Compute the coefficient of flow of a cycle signature.
        
        Parameters:
        cycle_signature (str): The cycle signature.

        Returns:
        float: The coefficient of flow.
        """
        link_flows = IFN.signature2link_flow(cycle_signature)
        kappa = IFN.signature2_kappa(cycle_signature)
        avg_flow = kappa / len(link_flows)
        variance = sum((flow - avg_flow) ** 2 for flow in link_flows.values()) / len(link_flows)
        return (variance ** 0.5) / avg_flow

    @staticmethod
    def signature2_max_flow(cycle_signature):
        """
        Compute the maximum flow value in a cycle signature.
        
        Parameters:
        cycle_signature (str): The cycle signature.

        Returns:
        float: The maximum flow value.
        """
        adjL = IFN.signature2_ideal_flow(cycle_signature)
        return max(link_flow for node in adjL for link_flow in adjL[node].values())

    @staticmethod
    def signature2_min_flow(cycle_signature):
        """
        Compute the minimum flow value in a cycle signature.
        
        Parameters:
        cycle_signature (str): The cycle signature.

        Returns:
        float: The minimum flow value.
        """
        adjL = IFN.signature2_ideal_flow(cycle_signature)
        return min(link_flow for node in adjL for link_flow in adjL[node].values())

    @staticmethod
    def signature2_sum_rows(cycle_signature):
        """
        Compute the sum of rows in a cycle signature.
        
        Parameters:
        cycle_signature (str): The cycle signature.

        Returns:
        dict: The row sums.
        """
        terms = IFN._parse_terms_to_dict(cycle_signature)
        row_sums = {}
        for cycle, coef in terms.items():
            for node in cycle:
                row_sums[node] = row_sums.get(node, 0) + coef
        return row_sums

    @staticmethod
    def signature2_sum_cols(cycle_signature):
        """
        Compute the sum of columns in a cycle signature.
        
        Parameters:
        cycle_signature (str): The cycle signature.

        Returns:
        dict: The column sums.
        """
        return IFN.signature2_sum_rows(cycle_signature)

    @staticmethod
    def signature2_pivots(cycle_signature):
        """
        Find pivots in a cycle signature.
        
        Parameters:
        cycle_signature (str): The cycle signature.

        Returns:
        dict: The pivots.
        """
        terms = IFN._parse_terms_to_dict(cycle_signature)
        cycles = list(terms.keys())
        pivots = {}
        for i, cycle in enumerate(cycles):
            for j in range(i + 1, len(cycles)):
                common_nodes = IFN._find_common_nodes(cycle, cycles[j])
                if common_nodes:
                    pivots[f"{cycle}-{cycles[j]}"] = common_nodes
        return pivots

    @staticmethod
    def is_irreducible_signature(cycle_signature):
        """
        Check if a cycle signature is irreducible.
        
        Parameters:
        cycle_signature (str): The cycle signature.

        Returns:
        bool: True if the signature is irreducible, otherwise False.
        """
        terms = IFN._parse_terms_to_dict(cycle_signature)
        cycles = list(terms.keys())
        for i, cycle in enumerate(cycles):
            if not any(IFN._find_common_nodes(cycle, cycles[j]) for j in range(len(cycles)) if i != j):
                return False
        return True

    @staticmethod
    def find_ifn_signature(num_nodes=4, kappa=17):
        """
        Generate a random IFN signature.
        
        Parameters:
        num_nodes (int): The number of nodes.
        kappa (int): The kappa value.

        Returns:
        str: The generated IFN signature.
        """
        nodes = [chr(97 + i) for i in range(num_nodes)]
        first_cycle = ''.join(nodes)
        first_cycle_length = len(first_cycle)
        cycle_dict = {first_cycle: 1}
        remaining_kappa = kappa - first_cycle_length
        count_loop = 0
        while remaining_kappa > 0:
            existing_cycles = list(cycle_dict.keys())
            random_cycle = existing_cycles[np.random.randint(len(existing_cycles))]
            pivot_start = np.random.randint(len(random_cycle))
            pivot_length = np.random.randint(1, len(random_cycle) - pivot_start + 1)
            pivot = random_cycle[pivot_start:pivot_start + pivot_length]
            new_cycle_nodes = set(pivot)
            while len(new_cycle_nodes) < pivot_length + np.random.randint(0, num_nodes - pivot_length + 1):
                new_cycle_nodes.add(nodes[np.random.randint(num_nodes)])
            if len(new_cycle_nodes) == 1 and count_loop < 100:
                count_loop += 1
                continue
            new_cycle = IFN.canonize(list(new_cycle_nodes))
            new_cycle_length = len(new_cycle)
            new_cycle_coefficient = remaining_kappa // new_cycle_length
            if new_cycle_coefficient > 0:
                cycle_dict[new_cycle] = cycle_dict.get(new_cycle, 0) + new_cycle_coefficient
                remaining_kappa -= new_cycle_coefficient * new_cycle_length
            elif new_cycle_length <= remaining_kappa:
                cycle_dict[new_cycle] = cycle_dict.get(new_cycle, 0) + 1
                remaining_kappa -= new_cycle_length
        return IFN.cycle_dict2_signature(cycle_dict)

    @staticmethod
    def cardinal_ifn(A):
        """
        Find the cardinal IFN signature.
        
        Parameters:
        A (list of list of int/float): The input matrix.

        Returns:
        str: The cardinal IFN signature.
        """
        all_cycles = IFN.find_all_cycles_in_matrix(A)
        sorted_cycles = sorted(all_cycles, key=len, reverse=True)
        lookup_set = IFN.signature2_links(' + '.join(all_cycles))
        selected_cycles = []
        current_links = set()
        for cycle in sorted_cycles:
            cycle_links = IFN.signature2_links(cycle)
            new_links = cycle_links - current_links
            if new_links:
                selected_cycles.append(cycle)
                current_links |= new_links
        selected_signature = ' + '.join(selected_cycles)
        return selected_signature if current_links == lookup_set else None

    @staticmethod
    def find_cardinal_ifn_exhaustive(A):
        """
        Find the cardinal IFN signature using exhaustive search.
        
        Parameters:
        A (list of list of int/float): The input matrix.

        Returns:
        str: The cardinal IFN signature.
        """
        all_cycles = IFN.find_all_cycles_in_matrix(A)
        all_combinations = IFN.generate_combinations(all_cycles)
        lookup_set = IFN.signature2_links(' + '.join(all_cycles))
        min_flow = float('inf')
        cardinal_signature = None
        for combination in all_combinations:
            signature = ' + '.join(combination)
            links = IFN.signature2_links(signature)
            if links == lookup_set:
                flow = IFN.signature2_kappa(signature)
                if flow < min_flow:
                    min_flow = flow
                    cardinal_signature = signature
        return cardinal_signature

    @staticmethod
    def assign2adjL(value, node_sequence, is_cycle=True):
        """
        Assign a value to a sequence of nodes.
        operator in AdjList
        
        Parameters:
        value (int/float): The value to assign.
        node_sequence (str): The sequence of nodes.
        is_cycle (bool): Whether the sequence represents a cycle.

        Returns:
        dict: The adjacency list.
        """
        nodes = node_sequence
        adj_list = {}
        for i in range(len(nodes)):
            current = nodes[i]
            next_ = nodes[(i + 1) % len(nodes)] if is_cycle else nodes[i + 1] if i + 1 < len(nodes) else None
            if next_:
                if current not in adj_list:
                    adj_list[current] = {}
                adj_list[current][next_] = adj_list[current].get(next_, 0) + value
        return adj_list

    @staticmethod
    def merge2adjL(adj_list1, adj_list2):
        """
        Merge two adjacency lists.
        operator in AdjList
        
        Parameters:
        adj_list1 (dict): The first adjacency list.
        adj_list2 (dict): The second adjacency list.

        Returns:
        dict: The merged adjacency list.
        """
        merged_adj_list = adj_list1.copy()
        for node, targets in adj_list2.items():
            if node not in merged_adj_list:
                merged_adj_list[node] = {}
            for target, value in targets.items():
                merged_adj_list[node][target] = merged_adj_list[node].get(target, 0) + value
        return merged_adj_list

    @staticmethod
    def merge_signatures(sig1, sig2):
        """
        Merge two network signatures.
        operator in network signature

        Parameters:
        sig1 (str): The first signature.
        sig2 (str): The second signature.

        Returns:
        str: The merged signature.
        """
        if not sig1:
            return sig2
        if not sig2:
            return sig1
        return f"{sig1} + {sig2}"

    @staticmethod
    def signature2_adj_list(signature, is_cycle=True):
        """
        Convert network signature to adjacency list.
        
        Parameters:
        signature (str): The network signature.
        is_cycle (bool): Whether the signature represents a cycle.

        Returns:
        dict: The adjacency list.
        """
        adj_list = {}
        terms = IFN._parse_terms_to_dict(signature)
        for cycle, coef in terms.items():
            partial_adj_list = IFN.assign2adjL(coef, cycle, is_cycle)
            adj_list = IFN.merge2adjL(adj_list, partial_adj_list)
        return adj_list



    '''
        PRIVATE PROCEDURES
    '''
    
    def __updateNetworkProbability__(self):
        """
        update the value of self.networkProb
        """
        kappa=self.totalFlow() 
        if kappa>0:
            adjList=self.__copyDict__(self.adjList)
            updatedAdjList=self.__updateAdjList__(adjList,1/kappa)
            self.networkProb=updatedAdjList
        

    def __updateAdjList__(self,adjList,factor):
        """
        generic multiplication any adjList weight with a scalar factor
        equivalent to matrix scalar multiple
        """
        updatedAdjList={}
        for startNode in adjList.keys(): 
            toNodes=adjList[startNode]
            for endNode,weight in toNodes.items():
                toNodes[endNode]=weight*factor
            updatedAdjList[startNode]=toNodes
        return updatedAdjList
    
    
    def __adjList2Matrix__(self,adjList):
        """
        generic conversion any adjacency list to adjacency matrix, listNode
        """
        listNode=self.__adjList2listNode__(adjList)
        n=len(listNode)
        matrix=self.__nullMatrix__(n,n)
        for row,startNode in enumerate(listNode):
            toNodes=self.outNeighbors(startNode)
            for endNode,weight in toNodes.items():
                col=listNode.index(endNode)
                matrix[row][col]=weight
        return matrix, listNode
    
    
    def __matrix2AdjList__(self, matrix,listNode):
        """
        return generic adjacency list from any adjacency matrix & listNode to 
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
    
    
    def __adjList2listNode__(self,adjList):
        """
        return generic listNode from any adjacency list
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
    
    
    def __getLinkWeight__(self,startNode,endNode,adjList):
        """
        return generic weight of a link [start,end] for any adjList
        """
        try:
            toNodes=adjList[startNode]
            weight=toNodes[endNode]
        except:
            weight=0 # if link does not exist
        return weight
    
    
    def __getWeightLink__(self,startNode,endNode):
        """
        return weight of link [start,end] of this internal adjList
        """
        weight=0
        toNodes=self.outNeighbors(startNode)
        if endNode in toNodes:
            weight=toNodes[endNode]
        return  weight
    
    
    def __setWeightLink__(self,startNode,endNode,weight):
        """
        set weight of a link directly
        if not exist, create the link and node
        """
        # add startNode and endNode if not exist
        if startNode not in self.listNodes:
            self.addNode(startNode)
        if endNode not in self.listNodes:
            self.addNode(endNode)
            
        if startNode in self.adjList.keys():
            # if startNode exists in adjList
            toNodes=self.adjList[startNode]
            toNodes[endNode]=weight
        else: # if startNode is not yet in adjList
            # create this endNode with weight
            toNodes={endNode: weight}
        self.adjList[startNode]=toNodes
    
    
    def __sumDictValues__(self,dic):
        """
        return generic sum of dictionary value
        """
        s=0
        for w in dic.values():
            if w==None:
                w=0
            s=s+w
        return s
        
    @staticmethod
    def __infMatrix__(mR,mC):
        """
        return matrix of 0 size (mR,mC)
        """
        return [[math.inf for i in range(mC)] for j in range(mR)]
    
    @staticmethod
    def __nullMatrix__(mR,mC):
        """
        return matrix of 0 size (mR,mC)
        """
        return [[0 for i in range(mC)] for j in range(mR)]
    
    @staticmethod
    def __matrixReplaceValue__(m,oldValue,newValue):
        """
        return matrix with old value replace by new value
        """
        return [[newValue if x == oldValue else x for x in row] for row in m]
    
    @staticmethod
    def __inverseDict__(dic):
        """
        return inverse dictionary from key:val to val:[keys]

        Example:
        # dic = {'a':1, 'b': 2} --> invDic = {1:['a'], 2:['b']}
        # dic = {'a': 3, 'c': 2, 'b': 2, 'e': 3, 'd': 1, 'f': 2}
        --> invDic =
                {1: ['d'], 2: ['c', 'b', 'f'], 3: ['a', 'e']}
        """
        invDic = {}
        for k, v in dic.items():
            invDic.setdefault(v, []).append(k)
        return invDic
    
    @staticmethod
    def __copyDict__(dic):
        """
        copy nested dictionary, list, string, number
        Note: it has security implication but works fast
        https://stackoverflow.com/questions/7845152/deep-copy-nested-list-without-using-the-deepcopy-function
        """
        return eval(repr(dic))
        # cpDic = {}
        # for k, v in dic.items():
        #     cpDic[k]=v
        # return cpDic
    
    
    @staticmethod
    def __num_to_excel_col__(num):
        """
        # return a, b, ..., z, aa, ab, ..., az, ba, ...
        # we use it to rename the variable
        # https://stackoverflow.com/questions/42176498/repeating-letters-like-excel-columns
        """
        if num < 1:
            num=1 
        result = ""
        while True:
            if num > 26:
                num, r = divmod(num - 1, 26)
                result = chr(r + ord('a')) + result
            else:
                return chr(num + ord('a') - 1) + result
    
    @staticmethod
    def excel_col_to_num(col):
        """
        Convert Excel column label to number.
        """
        num = 0
        for c in col:
            num = num * 26 + (ord(c.upper()) - ord('A') + 1)
        return num

    @staticmethod
    def __findElementInList__(element, listElement): 
        """
        return all indexes of element in the list if element is found else returns []
        """ 
        indexes = [i for i,x in enumerate(listElement) if x == element]
        return indexes
    
    @staticmethod
    def __findKeyInDic__(val,dic):
        """
        return  first matching key in a dictionary given a value
        source: https://stackoverflow.com/questions/16588328/return-key-by-value-in-dictionary
        """
        return next((k for k, v in dic.items() if v == val), None)
        # for k, v in dic.items():
        #     if v == val:
        #         return k
        # return None

    
    
    def __colorGraph(self,M, color, pos, c): 
        """
        # true if adjacency Matrix can be colored into two colors
        # called by isBipartite()
        # https://www.geeksforgeeks.org/bipartite-graph/
        """
        V=self.totalNodes()
        if color[pos] != -1 and color[pos] != c:  
            return False 
              
        # color this pos as c and all its neighbours and 1-c  
        color[pos] = c  
        ans = True 
        for i in range(0, V-1):
            if M[pos][i]:  
                if color[i] == -1:  
                    ans &= self.__colorGraph(M, color, i, 1-c)  
                      
                if color[i] !=-1 and color[i] != 1-c:  
                    return False 
               
            if not ans:  
                return False 
           
        return True 
    
    
    @staticmethod
    def decimal_to_fraction(decimal, tolerance=1.0E-6):
        """
        Converts a decimal to its closest fraction using a form of the Stern-Brocot tree approach.

        Parameters:
        decimal (float): The decimal number to convert.
        tolerance (float): Precision tolerance.

        Returns:
        tuple: The numerator and denominator of the fraction.
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
    def num_to_str_fraction(num):
        """
        Converts a number to a string fraction representation.

        Parameters:
        num (float): The number to convert.

        Returns:
        str: The string representation of the fraction.
        """
        if num == 0:
            return str(num)
        else:
            n, d = IFN.decimal_to_fraction(num)
            if d == 1:
                return str(n)
            else:
                return f"{n}/{d}"

    @staticmethod
    def combinations(N):
        """
        Generates all combinations of the letters 'a' through 'z' up to length N.

        Parameters:
        N (int): The length of the desired combinations.

        Returns:
        list of str: The list of combinations.
        """
        results = []
        letters = 'abcdefghijklmnopqrstuvwxyz'[:N]

        def combine(temp, start):
            if temp:
                results.append(''.join(temp))
            for i in range(start, len(letters)):
                temp.append(letters[i])
                combine(temp, i + 1)
                temp.pop()

        combine([], 0)
        return results

    @staticmethod
    def permutations(N):
        """
        Generates all non-empty permutations of N letters.

        Parameters:
        N (int): The number of letters to permute.

        Returns:
        list of str: The list of permutations.
        """
        results = []
        letters = 'abcdefghijklmnopqrstuvwxyz'[:N]

        def permute(arr, l, r):
            if l == r:
                results.append(''.join(arr))
            else:
                for i in range(l, r + 1):
                    arr[l], arr[i] = arr[i], arr[l]
                    permute(arr, l + 1, r)
                    arr[l], arr[i] = arr[i], arr[l]

        permute(list(letters), 0, N - 1)
        return results

    @staticmethod
    def generate_combinations(elements):
        """
        Generates all combinations of a given list of elements.

        Parameters:
        elements (list): The list of elements.

        Returns:
        list of list: The list of combinations.
        """
        result = []

        def f(prefix, elements):
            for i in range(len(elements)):
                result.append(prefix + [elements[i]])
                f(prefix + [elements[i]], elements[i + 1:])

        f([], elements)
        return result

    '''
    working code from Cycle6
    '''
        
    @staticmethod
    def from_base62(s):
        chars = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        base = 62
        num = 0
        for char in s:
            num = num * base + chars.index(char)
        return num

    @staticmethod
    def to_base62(num):
        chars = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        base = 62
        result = []
        while num > 0:
            result.append(chars[num % base])
            num //= base
        return ''.join(reversed(result)) or '0'

    @staticmethod
    def isPremier(cycle_str):
        parts = cycle_str.split('+')
        for part in parts:
            count_str = ''.join(filter(str.isdigit, part))
            count = int(count_str) if count_str else 1
            if count != 1:
                return False
        return True

    @staticmethod
    def node_index(name):
        if len(name) == 1 and 'a' <= name <= 'z':
            return ord(name) - 97
        else:
            return IFN.from_base62(name) + 26

    @staticmethod
    def node_name(idx):
        if idx < 26:
            return chr(idx + 97)  # 'a' to 'z' for the first 26 nodes
        else:
            return IFN.to_base62(idx - 26)  # Base62 encoding for the rest

    @staticmethod
    def find_cycles(matrix):
        n = len(matrix)
        cycles = set()

        # Convert adjacency matrix to adjacency list
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
    def canonical_cycle(cycle, node_mapping):
        mapped_cycle = [node_mapping[node] for node in cycle]
        min_idx = min(range(len(mapped_cycle)), key=lambda i: mapped_cycle[i:] + mapped_cycle[:i])
        rotated = mapped_cycle[min_idx:] + mapped_cycle[:min_idx]
        reverse = rotated[::-1]
        return min(''.join(rotated), ''.join(reverse))

    @staticmethod
    def canonicalize_cycle_string(cycle_string):
        unique_nodes = IFN.identify_unique_nodes(cycle_string)
        node_mapping = {node: IFN.node_name(i) for i, node in enumerate(unique_nodes)}
        # reverse_mapping = {v: k for k, v in node_mapping.items()}
        
        parts = cycle_string.split('+')
        canonical_parts = []
        for part in parts:
            count_str = ''.join(filter(str.isdigit, part))
            count = int(count_str) if count_str else 1
            cycle_str = ''.join(filter(str.isalpha, part))
            cycle = IFN.parse_cycle(cycle_str)
            # canonical_cycle_str = canonical_cycle([reverse_mapping[node] for node in cycle_str], node_mapping)
            canonical_cycle_str = IFN.canonical_cycle([IFN.node_name(IFN.parse_cycle(node)[0]) for node in cycle_str], node_mapping)
            if count == 1:
                canonical_parts.append(canonical_cycle_str)
            else:
                canonical_parts.append(f"{count}{canonical_cycle_str}")
        return '+'.join(canonical_parts)


    # Form the link-cycle matrix H and the link flow vector y
    @staticmethod
    def form_link_cycle_matrix(F):
        n = len(F)
        A = (F != 0).astype(int)
        cycles = IFN.find_cycles(A)

        links = [(i, j) for i in range(n) for j in range(n) if F[i][j] != 0]
        H = np.zeros((len(links), len(cycles)), dtype=int)

        for link_idx, (i, j) in enumerate(links):
            for cycle_idx, cycle in enumerate(cycles):
                cycle_nodes = [ord(c) - 97 if 'a' <= c <= 'z' else IFN.from_base62(c) + 26 for c in cycle]
                if i in cycle_nodes and j in cycle_nodes:
                    pos_i = cycle_nodes.index(i)
                    if cycle_nodes[(pos_i + 1) % len(cycle_nodes)] == j:
                        H[link_idx][cycle_idx] = 1

        y = F[F != 0]
        return H, y, cycles, links

    @staticmethod
    def parse_cycle(cycle):
        def node_index(name):
            if len(name) == 1 and 'a' <= name <= 'z':
                return ord(name) - 97
            else:
                return IFN.from_base62(name) + 26
        return [node_index(name) for name in cycle]

    @staticmethod
    def identify_unique_nodes(cycle_string):
        unique_nodes = set()
        parts = cycle_string.split('+')
        for part in parts:
            cycle_str = ''.join(filter(str.isalpha, part))
            unique_nodes.update(cycle_str)
        return sorted(unique_nodes)

    @staticmethod
    def assign_cycle(F, cycle, count):
        for i in range(len(cycle) - 1):
            F[cycle[i]][cycle[i + 1]] += count
        F[cycle[-1]][cycle[0]] += count  # Close the cycle

    @staticmethod
    def string_to_matrix(cycle_string):
        unique_nodes = IFN.identify_unique_nodes(cycle_string)
        node_mapping = {node: i for i, node in enumerate(unique_nodes)}
        
        n = len(unique_nodes)
        F = np.zeros((n, n), dtype=int)
        parts = cycle_string.split('+')
        for part in parts:
            count_str = ''.join(filter(str.isdigit, part.strip()))
            count = int(count_str) if count_str else 1
            cycle_str = ''.join(filter(str.isalpha, part.strip()))
            cycle = [node_mapping[node] for node in cycle_str]
            IFN.assign_cycle(F, cycle, count)

        # parts = cycle_string.split(' + ')
        # for part in parts:
        #     count_str = ''.join(filter(str.isdigit, part))
        #     count = int(count_str) if count_str else 1
        #     cycle_str = ''.join(filter(str.isalpha, part))
        #     cycle = [node_mapping[node] for node in cycle_str]
        #     assign_cycle(F, cycle, count)
        
        return F

    @staticmethod
    def solve_cycles(F):
        H, y, cycles, links = IFN.form_link_cycle_matrix(F)
        # method 1: can produce negative x[i]
        # x = np.dot(pinv(H), y)  # Solve Hx = y using the generalized inverse
        
        # method2: Solve Hx = y using least squares with non-negativity constraint
        result = np.lsq_linear(H, y, bounds=(0, np.inf))
        x = result.x

        # method 3: non negative
        # x, rnorm = nnls(H, y)  # Solve Hx = y using non-negative least squares
        
        # Normalize x to have integer values only if necessary
        min_nonzero = np.min(np.abs(x[np.nonzero(x)]))
        if not np.isclose(min_nonzero, round(min_nonzero)):
            x = np.round(x / min_nonzero).astype(int)
        else:
            x = np.round(x).astype(int)

        cycle_contributions = []
        for i, cycle in enumerate(cycles):
            if x[i] == 1:
                cycle_contributions.append(f"{cycle}")
            else:
                cycle_contributions.append(f"{x[i]}{cycle}")

        F_string = " + ".join(cycle_contributions)
        return F_string

    '''
        
            CYCLE-STRING 
        
    '''
    @staticmethod
    def extract_first_k_terms(cycle_str, k):
        parts = cycle_str.split(' + ')
        return ' + '.join(parts[:k])

    @staticmethod
    def extract_last_k_terms(cycle_str, k):
        parts = cycle_str.split(' + ')
        return ' + '.join(parts[-k:])

    @staticmethod
    def parse_terms_to_dict(cycle_str):
        terms = cycle_str.split(' + ')
        term_dict = {}
        for term in terms:
            count_str = ''.join(filter(str.isdigit, term))
            cycle_str = ''.join(filter(str.isalpha, term))
            if cycle_str not in term_dict:
                term_dict[cycle_str] = int(count_str) if count_str else 1
        return term_dict

    @staticmethod
    def generate_random_terms(cycle_dict, k, isPremier=False):
        terms = list(cycle_dict.keys())
        random.shuffle(terms)
        selected_terms = terms[:k]
        result_terms = []
        for term in selected_terms:
            if isPremier:
                result_terms.append(term)
            else:
                random_coefficient = random.randint(1, 10)  # Adjust range as needed
                if random_coefficient == 1:
                    result_terms.append(term)
                else:
                    result_terms.append(f"{random_coefficient}{term}")
        return ' + '.join(result_terms)

    @staticmethod
    def canonical_cycle(cycle, node_mapping):
            mapped_cycle = [node_mapping[node] for node in cycle]
            min_idx = min(range(len(mapped_cycle)), key=lambda i: mapped_cycle[i:] + mapped_cycle[:i])
            rotated = mapped_cycle[min_idx:] + mapped_cycle[:min_idx]
            reverse = rotated[::-1]
            return min(''.join(rotated), ''.join(reverse))

    @staticmethod
    def canonicalize_cycle_string(cycle_string):
        unique_nodes = IFN.identify_unique_nodes(cycle_string)
        node_mapping = {node: IFN.node_name(i) for i, node in enumerate(unique_nodes)}

        parts = cycle_string.split('+')
        canonical_parts = []
        for part in parts:
            count_str = ''.join(filter(str.isdigit, part.strip()))
            count = int(count_str) if count_str else 1
            cycle_str = ''.join(filter(str.isalpha, part.strip()))
            cycle = IFN.parse_cycle(cycle_str)
            canonical_cycle_str = IFN.canonical_cycle([IFN.node_name(node) for node in cycle], node_mapping)
            if count == 1:
                canonical_parts.append(canonical_cycle_str)
            else:
                canonical_parts.append(f"{count}{canonical_cycle_str}")
        return '+'.join(canonical_parts)

    @staticmethod
    def _node_name(idx):
        """
        Convert node index to node name.

        Parameters:
        idx (int): The node index.

        Returns:
        str: The node name.
        """
        return chr(97 + idx) if idx < 26 else IFN._to_base62(idx - 26)

    @staticmethod
    def _node_index(name):
        """
        Convert node name to node index.

        Parameters:
        name (str): The node name.

        Returns:
        int: The node index.
        """
        return ord(name) - 97 if name.islower() else IFN._from_base62(name) + 26

    @staticmethod
    def _from_base62(s):
        """
        Convert base62 string to number.

        Parameters:
        s (str): The base62 string.

        Returns:
        int: The number.
        """
        chars = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        return sum(chars.index(char) * (62 ** i) for i, char in enumerate(reversed(s)))

    @staticmethod
    def _to_base62(num):
        """
        Convert number to base62 string.

        Parameters:
        num (int): The number.

        Returns:
        str: The base62 string.
        """
        chars = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        result = ''
        while num > 0:
            result = chars[num % 62] + result
            num //= 62
        return result or '0'

    @staticmethod
    def min_irreducible(n):
        """
        Generate minimum irreducible matrix of size n.

        Parameters:
        n (int): The size of the matrix.

        Returns:
        list of list of int: The minimum irreducible matrix.
        """
        A = np.zeros((n, n), dtype=int)
        for r in range(n - 1):
            A[r][r + 1] = 1
        A[n - 1][0] = 1
        return A.tolist()

    @staticmethod
    def rand_int(mR, mC, max_val=10, prob=0.8):
        """
        Generate random integer matrix with biased zero entries.

        Parameters:
        mR (int): Number of rows.
        mC (int): Number of columns.
        max_val (int): Maximum value.
        prob (float): Probability of zeros.

        Returns:
        list of list of int: The random integer matrix.
        """
        m = np.zeros((mR, mC), dtype=int)
        for r in range(mR):
            for c in range(mC):
                random_num = random.randint(1, max_val)
                if random_num > max_val / 2 and random.random() < prob:
                    random_num = 0
                m[r][c] = random_num
        return m.tolist()

    @staticmethod
    def rand_stochastic(n):
        """
        Generate random stochastic matrix of size n.

        Parameters:
        n (int): Size of the matrix.

        Returns:
        list of list of float: The random stochastic matrix.
        """
        C = IFN.rand_int(n, n)
        mJ = np.ones((n, n))
        mDenom = C @ mJ
        S = C / mDenom
        return S.tolist()

    @staticmethod
    def rand_irreducible(num_nodes=5, num_links=8):
        """
        Generate random irreducible matrix.

        Parameters:
        num_nodes (int): Number of nodes.
        num_links (int): Number of links.

        Returns:
        list of list of int: The random irreducible matrix.
        """
        A = IFN.min_irreducible(num_nodes)
        A1 = IFN.add_random_ones(A, num_links)
        P = IFN.rand_permutation_eye(num_nodes)
        A2 = P @ A1 @ P.T
        return A2.tolist()

    @staticmethod
    def add_random_ones(A, m=6):
        """
        Add random ones to a matrix.

        Parameters:
        A (list of list of int): The input matrix.
        m (int): Number of ones to add.

        Returns:
        list of list of int: The updated matrix.
        """
        n = len(A)
        n2 = sum(sum(row) for row in A)
        if m > n2:
            k = 0
            while k < m - n2:
                idx = random.randint(0, n * n - 1)
                row = idx // n
                col = idx % n
                if A[row][col] == 0:
                    A[row][col] = 1
                    k += 1
        return A

    @staticmethod
    def rand_permutation_eye(n=5):
        """
        Generate random permutation of identity matrix.

        Parameters:
        n (int): Size of the matrix.

        Returns:
        list of list of int: The permuted identity matrix.
        """
        eye = np.eye(n, dtype=int)
        np.random.shuffle(eye)
        return eye.tolist()

    @staticmethod
    def is_irreducible_matrix(mA):
        """
        Check if a matrix is irreducible.

        Parameters:
        mA (list of list of int/float): The input matrix.

        Returns:
        bool: True if the matrix is irreducible, otherwise False.
        """
        mA = np.array(mA)
        n = mA.shape[0]
        eye = np.eye(n)
        mApI = mA + eye
        R = np.linalg.matrix_power(mApI, n - 1)
        return np.count_nonzero(R) == n * n

    @staticmethod
    def is_premagic_matrix(mA):
        """
        Check if a matrix is premagic.

        Parameters:
        mA (list of list of int/float): The input matrix.

        Returns:
        bool: True if the matrix is premagic, otherwise False.
        """
        mA = np.array(mA)
        n = mA.shape[0]
        if n != mA.shape[1]:
            return False
        vJ = np.ones((n, 1))
        vJt = np.ones((1, n))
        m1 = vJt @ mA
        m2 = (mA @ vJ).T
        return np.array_equal(m1, m2)

    @staticmethod
    def is_row_stochastic_matrix(mA):
        """
        Check if a matrix is row stochastic.

        Parameters:
        mA (list of list of int/float): The input matrix.

        Returns:
        bool: True if the matrix is row stochastic, otherwise False.
        """
        mA = np.array(mA)
        n = mA.shape[0]
        if n != mA.shape[1]:
            return False
        vJ = np.ones((n, 1))
        m1 = mA @ vJ
        return np.allclose(m1, vJ)

    @staticmethod
    def is_ideal_flow_matrix(mA):
        """
        Check if a matrix is an ideal flow matrix.

        Parameters:
        mA (list of list of int/float): The input matrix.

        Returns:
        bool: True if the matrix is an ideal flow matrix, otherwise False.
        """
        return IFN.is_premagic_matrix(mA) and IFN.is_irreducible_matrix(mA)

    @staticmethod
    def adj_list_to_matrix(adjL):
        """
        Convert an adjacency list to a weighted square matrix.

        Parameters:
        adjL (dict): The adjacency list.

        Returns:
        list of list of int: The weighted square matrix.
        """
        nodes = sorted(set(adjL.keys()).union(*(adjL[node].keys() for node in adjL)))
        size = len(nodes)
        node_index = {node: idx for idx, node in enumerate(nodes)}
        matrix = np.zeros((size, size), dtype=int)
        for node in adjL:
            for target in adjL[node]:
                i = node_index[node]
                j = node_index[target]
                matrix[i][j] = adjL[node][target]
        return matrix.tolist()

    @staticmethod
    def matrix_to_adj_list(matrix):
        """
        Convert a weighted square matrix to an adjacency list.

        Parameters:
        matrix (list of list of int/float): The weighted square matrix.

        Returns:
        dict: The adjacency list.
        """
        adj_list = {}
        size = len(matrix)
        for i in range(size):
            node = IFN._node_name(i)
            adj_list[node] = {}
            for j in range(size):
                if matrix[i][j] != 0:
                    adj_list[node][IFN._node_name(j)] = matrix[i][j]
        return adj_list

    @staticmethod
    def is_non_empty_adj_list(adjL):
        """
        Check if the adjacency list is non-empty.

        Parameters:
        adjL (dict): The adjacency list.

        Returns:
        bool: True if the adjacency list is non-empty, otherwise False.
        """
        return any(adjL[node] for node in adjL)

    @staticmethod
    def save_adj_list(adjL, filename):
        """
        Save an adjacency list to a file.

        Parameters:
        adjL (dict): The adjacency list.
        filename (str): The filename.

        Returns:
        None
        """
        with open(filename, 'w') as f:
            json.dump(adjL, f)

    @staticmethod
    def load_adj_list(filename):
        """
        Load an adjacency list from a file.

        Parameters:
        filename (str): The filename.

        Returns:
        dict: The adjacency list.
        """
        with open(filename, 'r') as f:
            return json.load(f)

    @staticmethod
    def find_a_cycle(start_node, target_node, adjL):
        """
        Find a cycle in the adjacency list starting from a given node.

        Parameters:
        start_node (str): The starting node.
        target_node (str): The target node to find the cycle.
        adjL (dict): The adjacency list.

        Returns:
        str: The cycle found or None if no cycle is found.
        """
        visited = set()
        stack = []

        def dfs(current_node):
            if current_node == target_node and stack:
                return ''.join(stack)
            visited.add(current_node)
            stack.append(current_node)
            for target in adjL.get(current_node, {}):
                if target not in visited or (target == target_node and len(stack) > 1):
                    cycle = dfs(target)
                    if cycle:
                        return cycle
            stack.pop()
            visited.remove(current_node)
            return None

        return dfs(start_node)

    @staticmethod
    def find_cycles(matrix):
        """
        Find all canonical cycles in a matrix.

        Parameters:
        matrix (list of list of int/float): The input matrix.

        Returns:
        list of str: The list of cycles.
        """
        n = len(matrix)
        cycles = set()

        def canonical(cycle):
            min_idx = min(range(len(cycle)), key=lambda i: cycle[i:] + cycle[:i])
            rotated = cycle[min_idx:] + cycle[:min_idx]
            reverse = rotated[::-1]
            return min(''.join(rotated), ''.join(reverse))

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
                        cycles.add(canonical(cycle))
            path.pop()
            visited[v] = False

        for i in range(n):
            visited = [False] * n
            path = []
            dfs(i, visited, path)

        return list(cycles)

    @staticmethod
    def find_all_permutation_cycles(matrix):
        """
        List all cycles in the adjacency matrix as strings.

        Parameters:
        matrix (list of list of int/float): The input matrix.

        Returns:
        set: The set of cycles.
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

    
if __name__=='__main__':
    net=IFN("abakadabra")
    plt.close('all')
    net.addLink("a","b",5)
    net.addLink("b","c",1)
    net.addLink("b","d",6)
    net.addLink("c","e",1)
    net.addLink("e","a",5)
    net.addLink("e","b",2)
    net.addLink("d","e",6)
    #net.unlearn(["b","d","a","b","e"])
    
    # net.show(layout=None)
    print(net)
    print('is idealflow?',net.isIdealFlow())
    C,listNodes=net.getMatrix()
    S=net.capacity2stochastic_proportional(C)
    F=net.Markov(S)
    net.applyMatrix(F,listNodes)
    print('is idealflow?',net.isIdealFlow())
    print(net.networkProb)
    net.__updateNetworkProbability__()
    print(net.networkProb)
    nodeSequence=["b","d","e","a","b"]
    print("pathEntropy1",net.getPathEntropy(nodeSequence))
    print(net.match(nodeSequence,{"abgh":net}),"\n")
    print("net",net,"\n")
    print("node values", net.nodesFlow())
    print('max link flow=',net.maxLinkFlow())
    
#     net=IFN()
#     k=5
#     m=k+int(3*k/4)
#     F=net.randIrreducible(k,m)
#     # print(F)
#     # net.applyMatrix(F,['a','b','c','d','#z#'])
#     net.applyMatrix(F,)
#     # print(net)
#     # traj=net.generate('#z#')
#     traj=net.generate('a')
#     print(traj)
#     trajSuper=net.orderMarkovHigher(traj,order=8)
#     print('trajSuper\n',trajSuper)
#     tr1=net.orderMarkovLower(trajSuper)
#     print('traj1\n',tr1)