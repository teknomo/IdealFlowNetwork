Data Structure
==========================

The structure of the IFN package is organized around two main components: **network directed graphs** and **matrices**. The dependencies between the packages are illustrated in the figure [here](packages_dependencies.jpg), and the class inheritance is shown in the figure [here](class_inheritances.jpg).

The IFN (Ideal Flow Network) package represents data using two primary formats: **Network Directed Graphs** and **Matrices**. These representations allow for flexible modeling of relationships between nodes and links in a network.

1. Network Directed Graph
-----------------------------------
- A **network** is represented as an **adjacency list**, which is a nested Python dictionary.
- A **node** is a key in the adjacency list, and each node is represented by its name (a string).
- **Links** between nodes are always directed, meaning they have a starting point and an endpoint. In the adjacency list:
  - The **starting node** is the key at the first level.
  - The **ending node** is the key at the second level.
- The **weight** of a link is stored as the value in the second level of the adjacency list. The weight can be an integer or a float (such as a probability).

**Example of a Directed Graph:**

.. code-block:: python

    {
        'a': {'b': 3, 'c': 1, 'd': 77},
        'b': {'c': 1, 'e': 1},
        'c': {'e': 1},
        'd': {'c': 1},
        'e': {'a': 1, 'f': 5}
    }


Another example using probabilities:

.. code-block:: python

    {
        'a': {'b': 0.012, 'c': 0.012, 'd': 0.916},
        'b': {'c': 0.012, 'e': 0.012},
        'c': {'e': 0.012},
        'd': {'c': 0.012},
        'e': {'a': 0.012}
    }

A real-world example using city names:

.. code-block:: python

    {
        'Calgary': {},
        'Chicago': {'Denver': 1000},
        'Denver': {'Houston': 1500, 'Los Angeles': 1000, 'Urbana': 1000},
        'Houston': {'Los Angeles': 1500},
        'Los Angeles': {},
        'New York': {'Chicago': 1000, 'Denver': 1900, 'Toronto': 800},
        'Toronto': {'Calgary': 1500, 'Chicago': 500, 'Los Angeles': 1800},
        'Urbana': {}
    }


2. Matrix Representation
-------------------------
- The **weighted adjacency matrix** is a two-dimensional array (a Python list of lists) that represents connections between nodes. Each element in the matrix represents the weight of the connection between nodes.
- The matrix is accompanied by a **list of nodes**, which represents the names of the nodes corresponding to the matrix rows and columns.

**Example of a Weighted Adjacency Matrix:**

.. code-block:: python

    matrix = [
        [0, 1, 1, 77, 0], 
        [0, 0, 1, 0, 1], 
        [0, 0, 0, 0, 1], 
        [0, 0, 1, 0, 0], 
        [1, 0, 0, 0, 0]
    ]
    nodes = ['a', 'b', 'c', 'd', 'e']


There are functions to convert between adjacency lists and matrices:
- To convert a matrix to an adjacency list, use the function :meth:`matrix_to_adj_list`.
- To convert an adjacency list to a matrix, use the function :meth:`adj_list_to_matrix`.

A **path** (or trajectory) through the network is represented as a sequence of node names (a Python list of strings).

**Example of a Path:**

.. code-block:: python

    ['a', 'b', 'e']


Naming Conventions
------------------------
We follow specific naming conventions in the IFN package to maintain consistency and readability:

- Each IFN has a **name** to represent the class.
- When saving an IFN, the **filename** is the same as the name of the IFN.
- **Private functions** start and end with double underscores (`__function_name__`).
- **Class names** start with a capital letter.
- **Function names** start with a lowercase letter.

### Abbreviations for Matrix Types:
- **A** = Adjacency matrix
- **B** = Incidence matrix
- **C** = Capacity matrix
- **F** = Flow matrix
- **S** = Stochastic matrix
- **sR** = Sum of rows
- **sC** = Sum of columns
- **kappa** = Total flow
- **pi** = Node vector (steady state)
- **[m, n]** = Matrix size (m rows, n columns)

Coding Standards
--------------------
For clarity and uniformity, we use the following terminologies across the IFN package:

- A **network** consists of **nodes** (vertices) and **links** (edges in graph theory).
- A **trajectory** (or path) is a sequence of nodes (or links).
- A **cycle** is a path where the start and end nodes are the same.
- **Flow** refers to the weight on a link, a node, or both.

As per our agreement, all private functions are named using double underscores (`__function_name__`) to clearly differentiate them from public methods.