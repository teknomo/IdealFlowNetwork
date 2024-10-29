Overview
============

The `IdealFlow` module provides tools for creating and managing Ideal Flow Networks (IFN). 
It includes functionalities for managing data, manipulating network structures, and 
querying network information. This module is designed to be flexible and scalable for 
various network analysis applications.

Main Classes
-----------------------
The `IdealFlow` module includes the following primary class:

- **IFN**: The main class representing an Ideal Flow Network, which provides various methods for analyzing and manipulating nodes, links, adjacency lists, matrices, 
    and performing network analysis and metric, including path finding, cycles analysis, analyzing connectivity, signature, and more.
    This class is designed to handle flow networks and their properties through mathematical operations and context.   


Example Usage
-----------------
Here is a basic example of how to use the `IdealFlow` module:

.. code-block:: python

    from IdealFlow.Network import IFN

    # Create a new Ideal Flow Network instance
    n = IFN()

    # Set up the adjacency list
    adj_list = {'a': {'b': 2}, 'b': {'c': 2}, 'c': {'a': 1, 'd': 1}, 'd': {'e': 1}, 'e': {'a': 1}}
    n.set_data(adj_list)

    # Add a link between nodes 'a' and 'b'
    n.add_link('a', 'b', 5)

    # Query the neighbors of node 'a'
    in_neighbors = n.in_neighbors('a')
    print("In-Neighbors of 'a':", in_neighbors)
    out_neighbors = n.out_neighbors('a')
    print("Out-Neighbors of 'a':", out_neighbors)

    # Display the flows
    n.show();

The above example demonstrates how to create a network, add links, and retrieve network information.

Categories
---------------

For more detailed information, see the following sections:

* `Documentation Index <index.html>`_

* `Overview <overview.html>`_

* `Beginner's Guide <BeginnerGuide.html>`_

* `Tutorial <tutorial.html>`_

* `API Reference <modules.html>`_

* `Node Management <node_management.html>`_

* `Link Management <link_management.html>`_   

* `Flow_management <flow_management.html>`_ 

*  `Adjacency List <adjList.html>`_

*  `Path Analysis  <path.html>`_

*  `Network Indices <network_indices.html>`_

*  `Network Operation Methods <network_operation.html>`_

*  `Matrix   <matrix.html>`_

*  `Stochastic  <stochastic.html>`_

*  `Markov Chain <markov.html>`_
   
* `Query Methods <query_methods.html>`_

*  `Neighborhood  <neighborhood.html>`_

*  `Ideal Flow Analysis   <ifn.html>`_

*  `Random Walk  <random_walk.html>`_

*  `Entropy <entropy.html>`_

*  `Cycle Analysis <cycle_analysis.html>`_

*  `Signature Analysis  <signature_analysis.html>`_
   
*  `Utility <utility.html>`_

*  `Markov Order <MarkovOrder.html>`_

* `Data Management Methods <data_management.html>`_


Applications Layer
-------------------

*  `Data Science, Artificial Intelligence, Machine Learning  <data_science_ai.html>`_

*  `Transportation Analysis <transportation_analysis.html>`_
