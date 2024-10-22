Data Structure
==========================

IFN is basically based on two kind of representations:

1.	Network Directed Graph
-----------------------------------

* A network is represented as Adjacency List, which is Python nested dictionary
* A node is represented by a string name in the key of adjacency list
* A link is always directed, the start node is key of the first level of the adjacency list and the end node is the key of the second level of the adjacency list.
* The weight in the value of the second level of the adjacency list. The weight can be integer or float (e.g. probability)

Example:

>>>	{'a': {'b': 3, 'c': 1, 'd': 77}, 'b': {'c': 1, 'e': 1}, 'c': {'e': 1}, 'd': {'c': 1}, 'e': {'a': 1, 'f': 5}}


>>>	
	{'a': {'b': 0.012, 'c': 0.012, 'd': 0.916}, 
	'b': {'c': 0.012, 'e': 0.012}, 
	'c': {'e': 0.012}, 
	'd': {'c': 0.012}, 
	'e': {'a': 0.012}}


>>>	
	{'Calgary': {}, 'Chicago': {'Denver': 1000}, 
	'Denver': {'Houston': 1500, 'Los Angeles': 1000, 'Urbana': 1000},
	'Houston': {'Los Angeles': 1500}, 'Los Angeles': {}, 
	'New York': {'Chicago': 1000, 'Denver': 1900, 'Toronto': 800}, 
	'Toronto': {'Calgary': 1500, 'Chicago': 500, 'Los Angeles': 1800}, 'Urbana': {}}


2.	Matrix
-------------------

* Weighted Adjacency Matrix, which is Python two-dimensional array together with list nodes.

Example:

>>> 
	[[0, 1, 1, 77, 0], 
	[0, 0, 1, 0, 1], 
	[0, 0, 0, 0, 1], 
	[0, 0, 1, 0, 0], 
	[1, 0, 0, 0, 0]], 
	['a', 'b', 'c', 'd', 'e']
	

Function to convert matrix to adjacency list is :meth:`matrix_to_adj_list`.

Function to convert adjacency list to matrix is :meth:`adj_list_to_matrix`.

* A Path or a trajectory is represented as node sequence, which is a Python list of strings.

Example:

>>> ['a','b','e']



Naming Convention
-----------------------

* Name for each IFN to tell the class
* Saving IFN would be the same as the name
* All private functions would have name start and end with double underscore __.
* Class name start with capital letter
* Function name start with lower letter
* A = Adjacency matrix
* B = Incidence matrix
* C = Capacity matrix
* F = Flow matrix
* S = Stochastic matrix
* sR = sum of rows
* sC = sum of columns
* kappa = total flow
* pi = node vector (steady state)
* [m,n] = matrix size


Coding Standard
--------------------

As our agreements, we use the following terminologies 

* Network consists of nodes and links (= vertices and edges in graph theory)
* Trajectory = Path = node sequence = link sequence
* Cycle = path that have the same start and end 
* Flow = either link weight, node weight or both

All private functions would have name start and end with double underscore __.

