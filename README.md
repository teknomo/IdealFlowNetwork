
# Ideal Flow Network Python Library

[![PyPI version](https://badge.fury.io/py/IdealFlowNetwork.svg)](https://badge.fury.io/py/IdealFlowNetwork)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/Documentation-IdealFlowNetwork-brightgreen)](https://people.revoledu.com/kardi/research/trajectory/ifn/doc)
[![Documentation Status](https://readthedocs.org/projects/idealflownetwork/badge/?version=master)](https://idealflownetwork.readthedocs.io/?badge=master)
[![X Twitter](https://img.shields.io/twitter/follow/kteknomo?style=social)](https://x.com/kteknomo)
[![Telegram](https://img.shields.io/badge/Telegram-IdealFlowNetwork-blue.svg)](https://t.me/IdealFlowNetwork)
[![Discord](https://img.shields.io/badge/Discord-IdealFlowNetwork-7289da.svg)](https://discord.gg/fUVzBx5GF4)


The `IdealFlowNetwork` Python package provides tools for creating and managing **Ideal Flow Networks (IFN)**. An Ideal Flow Network is a strongly connected network where the flows are balanced. This concept is linked to how we think about honesty, integrity, and the long-term consequences of our actions, touching upon morality and spirituality. By understanding how systems work, we can make better decisions that align with these values.

## Table of Contents

- [Ideal Flow Network Python Library](#ideal-flow-network-python-library)
  - [Table of Contents](#table-of-contents)
  - [What is Ideal Flow Network?](#what-is-ideal-flow-network)
  - [Features](#features)
  - [Installation](#installation)
  - [Getting Started](#getting-started)
  - [Applications](#applications)
  - [Contributing](#contributing)
  - [Community and Support](#community-and-support)
  - [License](#license)
  - [Sponsors](#sponsors)
  - [Citing Ideal Flow Network](#citing-ideal-flow-network)
  - [Scientific Basis](#scientific-basis)
  - [Contact](#contact)

## What is Ideal Flow Network?

An **Ideal Flow Network (IFN)** is a theoretical model where the network is strongly connected, and the flows are balanced and conserved. Mathematically, it represents a steady-state relative flow distribution in a directed graph. The IFN theory was first proposed by [Kardi Teknomo](https://people.revoledu.com/kardi/Image/ExcellentPosterAwardWaseda2015.jpg) in 2015.

IFNs have profound implications in various fields, including transportation networks, communication networks, health science, agricultural ecology, data science, machine learning, artificial intelligence, systems thinking, philosophy, and more.

For a special application to transportation planning, check out the traffic assignment software [IFN-Transport](https://github.com/teknomo/ifn-transport).

## Features

- **Flexible and Scalable Modules**: Includes expandable Network, Classifier, Text, and Table modules designed for various network analysis applications.
- **Data Management**: Functions for managing data, manipulating network structures, and querying network information.
- **Comprehensive Analysis Tools**: Methods for analyzing and manipulating nodes, links, adjacency lists, matrices, and performing network analysis and metrics.
- **Advanced Algorithms**: Includes path finding, cycle analysis, connectivity analysis, signature computations, and more.

## Installation

You can install the package using pip:

```bash
pip install IdealFlowNetwork
```

Check out the package on [PyPI](https://pypi.org/project/IdealFlowNetwork/).

Latest Stable Version: [1.5.5](https://pypi.org/project/IdealFlowNetwork/1.5.5/)

The latest the zip file (still unstable) versioncan be downloaded from [GitHub](https://github.com/teknomo/IdealFlowNetwork), unzip the files in a folder. 
1. Go to folder *pkg*

```bash
cd *your_folder_name*`/pkg`
```
2. In command Prompt
```bash 
> pip install . 
```
Copy the package files to your Python environment. 

## Getting Started

Here's a simple example to get you started:

```python
import IdealFlow.Network as net

# Initialize an Ideal Flow Network
n = net.IFN()

# Build network simply by adding links or trajectories
n.add_link('a','b')
n.add_link('b','c')
n.add_link('c','a')
trajectory =  ['a','b','c','d','e','a']
n.assign(trajectory)

# Show the adjacency list
print(n)

# Extract matrix and list of nodes 
matrix, list_nodes = n.get_matrix()
print('matrix:',matrix)           # weighted adjacency matrix
print('list nodes:',list_nodes)           # weighted adjacency matrix

# Query the neighbors of node 'a'
in_neighbors = n.in_neighbors('a')
print("In-Neighbors of 'a':", in_neighbors)
out_neighbors = n.out_neighbors('a')
print("Out-Neighbors of 'a':", out_neighbors)

# Get network performances
print('Total Nodes:',n.total_nodes)           # number of nodes
print('Total Links:',n.total_links)           # number of links
print('List of Links:',n.get_links)           # list of  links
print('Total Flow:',n.total_flow)             # total flow
print('Network Entropy:',n.network_entropy)   # network entropy 
print('Coef. Variation of Flow:',n.cov_flow) # coef. variation of flow

# Display the flows
n.show();
```

For more detailed [tutorials](https://people.revoledu.com/kardi/tutorial/IFN/) and documentation, please visit our [official documentation](https://people.revoledu.com/kardi/research/trajectory/ifn/doc/html/).

## Applications

Ideal Flow Networks have been applied in various domains:

- **Transportation Planning**: Modeling and optimizing traffic flows.
- **Communication Networks**: Analyzing power dynamics of communications based on strength of influence and network structure.
- **Health Science**: Understanding the spread of diseases in epidemiology.
- **Ecology**: Studying energy and resource flows in ecosystems.
- **Data Science and AI**: Enhancing machine learning algorithms with network-based approaches.
- **Systems Thinking and Philosophy**: Exploring concepts of interconnectedness and systemic behavior.
- **Robot Process Automation (RPA)**: Create synchronous and synchronous automation based on Large Language Models (LLM), mouse recording and simulation, multi agent systems and many more based on Automation plugins.

## Contributing

We welcome contributions from everyone! If you're interested in contributing to the Ideal Flow Network project, here are some ways you can help:

- **Test**: Try out the package and let us know if you encounter any issues.
- **Comment and Critique**: Provide feedback to help us improve.
- **Use**: Incorporate IFN into your projects and share your experiences.
- **Post on Social Media**: Spread the word! Share our project on social media platforms.
- **Subscribe**: Stay updated by following our channels.

Feel free to pick an [issue](https://github.com/teknomo/IdealFlowNetwork/issues) and help us out! Pull requests are encouraged and always welcome. Check How to [Contributing to Ideal Flow Network](Contributing.md).

## Community and Support

Join our community to connect with other users and contributors:
- **Discord**: [IdealFlowNetwork](https://discord.gg/fUVzBx5GF4).
- **Telegram Channel**: [Ideal Flow Network](https://t.me/IdealFlowNetwork)
- **Twitter**: Follow us [@Revoledu](https://twitter.com/Revoledu)

## License

This project is licensed under the **GNU General Public License v3.0 (GPL-3.0)**. See the [LICENSE](LICENSE) file for details.

## Sponsors

We greatly appreciate any support to help us continue developing and improving this project. If you wish to support us and prefer a different licensing arrangement, we offer sponsorship opportunities.

**Note:** The farther away from GPL you want to be, the exponentially more support is required.

## Citing Ideal Flow Network

If you use **Ideal Flow Network** in your research or projects, please consider citing:

Teknomo, K. (2018). *Ideal Flow of Markov Chain*. Discrete Mathematics, Algorithms and Applications, 10(06), 1850073. doi: [10.1142/S1793830918500738](https://doi.org/10.1142/S1793830918500738)

Example BibTeX entry:

```bibtex
@article{doi:10.1142/S1793830918500738,
author = {Teknomo, Kardi},
title = {Ideal Flow of Markov Chain},
journal = {Discrete Mathematics, Algorithms and Applications},
volume = {10},
number = {06},
pages = {1850073},
year = {2018},
doi = {10.1142/S1793830918500738},
URL = {https://doi.org/10.1142/S1793830918500738},
}
```

You may also cite any of the publications listed in the [Scientific Basis](#scientific-basis) section if you use or improve this Python library.

## Scientific Basis

The following publications form the foundation of Ideal Flow Network analysis:

- Teknomo, K. (2019). [Ideal Flow Network in Society 5.0](https://link.springer.com/chapter/10.1007/978-3-030-28565-4_11). In *Optimization in Large Scale Problems - Industry 4.0 and Society 5.0 Applications*, Springer, pp. 67-69.
- Teknomo, K., & Gardon, R.W. (2019). [Traffic Assignment Based on Parsimonious Data: The Ideal Flow Network](https://ieeexplore.ieee.org/document/8917426). *2019 IEEE Intelligent Transportation Systems Conference (ITSC)*, 1393-1398.
- Teknomo, K., Gardon, R., & Saloma, C. (2019). [Ideal Flow Traffic Analysis: A Case Study on a Campus Road Network](https://philjournalsci.dost.gov.ph/images/pdf/pjs_pdf/vol148no1/ideal-flow-trappic-analysis_.pdf). *Philippine Journal of Science*, 148(1), 51-62.
- Teknomo, K. (2018). [Ideal Flow of Markov Chain](https://www.worldscientific.com/doi/pdf/10.1142/S1793830918500738). *Discrete Mathematics, Algorithms and Applications*, 10(06), 1850073.
- Teknomo, K., & Gardon, R.W. (2017). *Intersection Analysis Using the Ideal Flow Model*. Proceedings of the IEEE 20th International Conference on Intelligent Transportation Systems.
- Teknomo, K. (2017). *Ideal Relative Flow Distribution on Directed Network*. Proceedings of the 12th Eastern Asia Society for Transportation Studies (EASTS).
- Teknomo, K. (2017). [Premagic and Ideal Flow Matrices](https://arxiv.org/abs/1706.08856). arXiv preprint arXiv:1706.08856.
- Gardon, R.W., & Teknomo, K. (2017). *Analysis of the Distribution of Traffic Density Using the Ideal Flow Method and the Principle of Maximum Entropy*. Proceedings of the 17th Philippine Computing Science Congress.
- Teknomo, K. (2015). *Ideal Flow Based on Random Walk on Directed Graph*. The 9th International Collaboration Symposium on Information, Production and Systems (ISIPS 2015).

## Contact

(c) 2024 Kardi Teknomo

For any inquiries or support, please contact [Kardi Teknomo](http://people.revoledu.com/kardi/).