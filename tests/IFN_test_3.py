import unittest
import numpy as np
import random

"""
last update: Oct 14, 2024

IFN Unit Tests

@author: Kardi Teknomo
"""

import IdealFlow.Network as net         # import package.module as alias
import tempfile
import os

class TestIFN(unittest.TestCase):
    def setUp(self):
        self.ifn = net.IFN("TestNetwork")
        self.net = net.IFN("abakadabra")
        self.net.add_link("a", "b", 5)
        self.net.add_link("b", "c", 1)
        self.net.add_link("b", "d", 6)
        self.net.add_link("c", "e", 1)
        self.net.add_link("e", "a", 5)
        self.net.add_link("e", "b", 2)
        self.net.add_link("d", "e", 6)
    
    def test_initialization(self):
        self.assertEqual(self.ifn.version, "1.5.1")
        self.assertEqual(self.ifn.name, "TestNetwork")
        self.assertEqual(self.ifn.adjList, {})
        self.assertEqual(self.ifn.numNodes, 0)
        self.assertEqual(self.ifn.listNodes, [])
        self.assertEqual(self.ifn.network_prob, {})
        self.assertEqual(self.ifn.epsilon, 0.000001)
  
        
    
    def test_repr(self):
        self.assertEqual(repr(self.ifn), self.ifn.name)

    def test_str(self):
        self.assertEqual(str(self.ifn), "{}")

    def test_len(self):
        self.assertEqual(len(self.ifn), 0)

    def test_iter(self):
        self.assertEqual(list(iter(self.ifn)), [])

    def test_add_node(self):
        self.ifn.add_node("A")
        self.assertIn("A", self.ifn.listNodes)
        self.assertIn("A", self.ifn.adjList)
        self.assertEqual(self.ifn.numNodes, 1)

    def test_delete_node(self):
        self.ifn.add_node("A")
        self.ifn.delete_node("A")
        self.assertNotIn("A", self.ifn.listNodes)
        self.assertNotIn("A", self.ifn.adjList)
        self.assertEqual(self.ifn.numNodes, 0)

    def test_nodes_property(self):
        self.ifn.add_node("A")
        self.ifn.add_node("B")
        self.assertEqual(set(self.ifn.nodes), {"A", "B"})

    def test_total_nodes(self):
        self.ifn.add_node("A")
        self.ifn.add_node("B")
        self.assertEqual(self.ifn.total_nodes, 2)

    def test_nodes_flow(self):
        self.ifn.add_link("A", "B", 3)
        self.ifn.add_link("A", "C", 2)
        self.assertEqual(self.ifn.nodes_flow, {"A": 5, "B": 0, "C": 0})

    def test_add_link(self):
        self.ifn.add_link("A", "B", 5)
        self.assertIn("A", self.ifn.adjList)
        self.assertIn("B", self.ifn.adjList["A"])
        self.assertEqual(self.ifn.adjList["A"]["B"], 5)

    def test_delete_link(self):
        self.ifn.add_link("A", "B", 5)
        self.ifn.delete_link("A", "B")
        self.assertNotIn("B", self.ifn.adjList["A"])

    def test_set_link_weight(self):
        self.ifn.set_link_weight("A", "B", 7)
        self.assertEqual(self.ifn.adjList["A"]["B"], 7)

    def test_set_link_weight_plus1(self):
        self.ifn.set_link_weight_plus_1("A", "B")
        self.assertEqual(self.ifn.adjList["A"]["B"], 1)

    def test_get_links(self):
        self.ifn.add_link("A", "B", 5)
        self.ifn.add_link("A", "C", 3)
        self.assertEqual(self.ifn.get_links, [["A", "B"], ["A", "C"]])

    def test_signature_to_ideal_flow(self):
        result = self.net.signature_to_ideal_flow("abc")
        self.assertIsInstance(result, dict)
        
    def test_signature_to_kappa(self):
        kappa = self.net.signature_to_kappa("3abc")
        self.assertEqual(kappa, 9)
        
    def test_signature_to_coef_flow(self):
        coef_flow = self.net.signature_to_coef_flow("abc")
        self.assertEqual(coef_flow, 0)
        
    def test_signature_to_max_flow(self):
        max_flow = self.net.signature_to_max_flow("7abc")
        self.assertEqual(max_flow, 7)
        
    
    def test_signature_to_min_flow(self):
        min_flow = self.net.signature_to_min_flow("abc")
        self.assertEqual(min_flow, 1)
        
    def test_is_bipartite(self):
        matrix = [
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 0]
        ]
        self.net.set_matrix(matrix)
        self.assertTrue(self.net.is_bipartite)
        
    def test_get_link_flow(self):
        flow_values = self.net.get_link_flow("a","b")
        self.assertIsInstance(flow_values, (int, float))
        
    def test_get_matrix(self):
        matrix, nodes = self.net.get_matrix()
        self.assertIsInstance(matrix, list)
        self.assertIsInstance(nodes, list)
        
    def test_capacity_to_stochastic(self):
        matrix, _ = self.net.get_matrix()
        stochastic_matrix = self.net.capacity_to_stochastic(matrix)
        self.assertIsInstance(stochastic_matrix, np.ndarray)
        
    def test_markov(self):
        matrix, _ = self.net.get_matrix()
        stochastic_matrix = self.net.capacity_to_stochastic(matrix)
        markov_matrix = self.net.markov(stochastic_matrix)
        self.assertIsInstance(markov_matrix, np.ndarray)
        
    def test_set_matrix(self):
        matrix, nodes = self.net.get_matrix()
        stochastic_matrix = self.net.capacity_to_stochastic(matrix)
        markov_matrix = self.net.markov(stochastic_matrix)
        self.net.set_matrix(markov_matrix, nodes)
        self.assertIsInstance(self.net.network_prob, dict)

    def test_delete_link(self):
        self.ifn.add_link("A", "B", 5)
        self.ifn.delete_link("A", "B")
        if "A" in self.ifn.adjList:
            self.assertNotIn("B", self.ifn.adjList["A"])
        else:
            self.assertNotIn("A", self.ifn.adjList)

    def test_is_ideal_flow(self):
        self.assertIsInstance(self.net.is_ideal_flow, bool)
        
    def test___updateNetworkProbability__(self):
        self.net.__updateNetworkProbability__()
        self.assertIsInstance(self.net.network_prob, dict)
        
    def test_get_path_entropy(self):
        nodeSequence = ["b", "d", "e", "a", "b"]
        entropy = self.net.get_path_entropy(nodeSequence)
        self.assertIsInstance(entropy, (int, float))
        
    def test_match(self):
        nodeSequence = ["b", "d", "e", "a", "b"]
        match_result = self.net.match(nodeSequence, {"abgh": self.net})
        self.assertIsInstance(match_result, tuple)
        
    def test_nodes_flow(self):
        nodes_flow = self.net.nodes_flow()
        self.assertIsInstance(nodes_flow, dict)
        
    def test_max_link_flow(self):
        max_flow = self.net.max_link_flow()
        self.assertIsInstance(max_flow, (int, float))
        
    def test_generate(self):
        traj = self.net.generate('a')
        self.assertIsInstance(traj, list)
        
    def test_order_markov_higher(self):
        traj = self.net.generate('a')
        traj_super = self.net.order_markov_higher(traj, order=8)
        self.assertIsInstance(traj_super, list)
        
    def test_order_markov_lower(self):
        traj = self.net.generate('a')
        traj_super = self.net.order_markov_higher(traj, order=8)
        traj_lower = self.net.order_markov_lower(traj_super)
        self.assertIsInstance(traj_lower, list)
    
    def test_match(self):
        nodeSequence = ["b", "d", "e", "a", "b"]
        match_result = self.net.match(nodeSequence, {"abgh": self.net})
        self.assertIsInstance(match_result, tuple)
        
    def test_nodes_flow(self):
        nodes_flow = self.net.nodes_flow
        self.assertIsInstance(nodes_flow, dict)
        
    def test_max_link_flow(self):
        max_flow = self.net.max_flow
        self.assertIsInstance(max_flow, (int, float))
        
    def test_generate(self):
        traj = self.net.generate('a')
        self.assertIsInstance(traj, list)
        
    def test_order_markov_higher(self):
        traj = self.net.generate('a')
        traj_super = self.net.order_markov_higher(traj, order=8)
        self.assertIsInstance(traj_super, list)
        
    def test_order_markov_lower(self):
        traj = self.net.generate('a')
        traj_super = self.net.order_markov_higher(traj, order=8)
        traj_lower = self.net.order_markov_lower(traj_super)
        self.assertIsInstance(traj_lower, list)

    # Assuming the class has these methods as well:
    def test_add_link(self):
        self.net.add_link("f", "g", 3)
        matrix, nodes = self.net.get_matrix()
        self.assertIn("f", nodes)
        self.assertIn("g", nodes)

    def test_delete_link(self):
        self.net.delete_link("a", "b")
        matrix, nodes = self.net.get_matrix()
        self.assertNotIn("b", matrix[nodes.index("a")])

    def test_get_probability(self):
        prob=self.net.network_probability
        self.assertIsInstance(prob, (dict))


    def test_set_link_weight(self):
        self.net.set_link_weight("a", "b", 10)
        flow_values = self.net.get_link_flow("a", "b")
        self.assertEqual(flow_values, 10)
        
    def test__color_graph(self):
        M = [
            [0, 1, 1, 0],
            [1, 0, 1, 1],
            [1, 1, 0, 1],
            [0, 1, 1, 0]
        ]
        color = [-1, -1, -1, -1]
        result = self.net.color_graph(M, color, 0, 1)
        self.assertIsInstance(result, bool)
    
    def test_compose(self):
        signature = "abc"
        expected = [
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0]
        ]
        result = self.ifn.compose(signature)
        self.assertTrue(np.array_equal(result, expected))
        

    def test_decompose(self):
        F = [
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0]
        ]
        expected_signature = "abc"
        result = self.ifn.decompose(F)
        self.assertEqual(result, expected_signature)

    def test_flow_matrix_to_adj_list(self):
        F = [
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0]
        ]
        expected_adj_list = {
            0: {1: 1},
            1: {2: 1},
            2: {0: 1}
        }
        result = self.ifn.flow_matrix_to_adj_list(F)
        self.assertEqual(result, expected_adj_list)

    def test_decimal_to_fraction(self):
        decimal_number = 0.5
        expected_fraction = (5000, 10000)
        result = self.ifn.decimal_to_fraction(decimal_number)
        self.assertEqual(result, expected_fraction)

    def test_parse_terms_to_dict(self):
        signature = "2a + 3b + c"
        expected_dict = {
            "a": 2,
            "b": 3,
            "c": 1
        }
        result = self.ifn.parse_terms_to_dict(signature)
        self.assertEqual(result, expected_dict)

    def test_find_common_nodes(self):
        cycle1 = "abc"
        cycle2 = "bcd"
        expected_common_nodes = ["b", "c"]
        result = self.ifn._find_common_nodes(cycle1, cycle2)
        self.assertEqual(set(result), set(expected_common_nodes))

    # def test_canonical_cycle(self):
    #     cycle = ["b", "c", "a"]
    #     node_mapping = {"a": "a", "b": "b", "c": "c"}
    #     expected_canonical_cycle = ["a", "b", "c"]
    #     result = self.ifn._canonical_cycle(cycle, node_mapping)
    #     self.assertEqual(result, expected_canonical_cycle)

    def test_is_net_signature_premier(self):
        net_signature = "a + b + c"
        result = self.ifn.is_premier_signature(net_signature)
        self.assertTrue(result)

    def test_convert_signature_coef2one(self):
        net_signature = "2a + 3b + c"
        expected_signature = "a + b + c"
        result = self.ifn.signature_coef_to_1(net_signature)
        self.assertEqual(result, expected_signature)

    def test_parse_cycle(self):
        cycle = "abc"
        expected_indices = [0, 1, 2]
        result = self.ifn.parse_cycle(cycle)
        self.assertEqual(result, expected_indices)

    def test_assign_cycle(self):
        F = np.zeros((3, 3))
        cycle = [0, 1, 2]
        value = 1
        expected_matrix = [
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0]
        ]
        self.ifn.assign_cycle_to_matrix(F, cycle, value)
        self.assertEqual(F.tolist(), expected_matrix)

    def test_identify_unique_nodes(self):
        net_signature = "a + b + c"
        expected_nodes = ["a", "b", "c"]
        result = self.ifn.identify_unique_nodes(net_signature)
        self.assertEqual(result, expected_nodes)
    
    def test_decimal_to_fraction(self):
        self.assertEqual(self.ifn.decimal_to_fraction(0.5), (1, 2))
        self.assertEqual(self.ifn.decimal_to_fraction(0.3333333), (1, 3))

    def test_num_to_str_fraction(self):
        self.assertEqual(self.ifn.num_to_str_fraction(0), '0')
        self.assertEqual(self.ifn.num_to_str_fraction(0.5), '1/2')
        self.assertEqual(self.ifn.num_to_str_fraction(2), '2')

    def test_combinations(self):
        expected = ['a', 'b', 'c', 'ab', 'ac', 'bc', 'abc']
        result = self.ifn.combinations(3)
        self.assertEqual(sorted(result), sorted(expected))

    def test_permutations(self):
        expected = ['a', 'b', 'c', 'ab', 'ac', 'ba', 'bc', 'ca', 'cb', 'abc', 'acb', 'bac', 'bca', 'cab', 'cba']
        result = self.ifn.permutations(3)
        self.assertEqual(sorted(result), sorted(expected))

    def test_generate_combinations(self):
        elements = ['a', 'b', 'c']
        expected = [['a'], ['a', 'b'], ['a', 'b', 'c'], ['a', 'c'], ['b'], ['b', 'c'], ['c']]
        result = self.ifn.generate_combinations(elements)
        self.assertEqual(sorted(result), sorted(expected))
    
    def testnode_name(self):
        self.assertEqual(self.ifn.node_name(0), 'a')
        self.assertEqual(self.ifn.node_name(25), 'z')
        self.assertEqual(self.ifn.node_name(26), 'A')
        self.assertEqual(self.ifn.node_name(114), '10')
        self.assertEqual(self.ifn.node_name(682), 'aa')

    def testnode_index(self):
        self.assertEqual(self.ifn.node_index('a'), 0)
        self.assertEqual(self.ifn.node_index('z'), 25)
        self.assertEqual(self.ifn.node_index('A'), 26)
        self.assertEqual(self.ifn.node_index('10'), 114)
        self.assertEqual(self.ifn.node_index('aa'), 682)

    def testfrom_base62(self):
        self.assertEqual(self.ifn.from_base62('0'), 0)
        self.assertEqual(self.ifn.from_base62('A'), 36)
        self.assertEqual(self.ifn.from_base62('10'), 62)
        self.assertEqual(self.ifn.from_base62('1Z'), 123)

    def testto_base62(self):
        self.assertEqual(self.ifn.to_base62(62), '10')
        self.assertEqual(self.ifn.to_base62(123), '1Z')

    def test_min_irreducible(self):
        expected = [
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0]
        ]
        result = self.ifn.min_irreducible(3)
        self.assertTrue(np.array_equal(result, expected))
        

    def test_rand_int(self):
        result = self.ifn.rand_int(3, 3)
        self.assertEqual(len(result), 3)
        self.assertEqual(len(result[0]), 3)

    def test_rand_stochastic(self):
        result = self.ifn.rand_stochastic(3)
        self.assertEqual(len(result), 3)
        self.assertEqual(len(result[0]), 3)

    def test_rand_irreducible(self):
        result = self.ifn.rand_irreducible(3, 4)
        self.assertEqual(len(result), 3)
        self.assertEqual(len(result[0]), 3)

    def test_add_random_ones(self):
        matrix = [
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0]
        ]
        result = self.ifn.add_random_ones(matrix, 5)
        self.assertEqual(sum(sum(row) for row in result), 5)

    def test_rand_permutation_eye(self):
        result = self.ifn.rand_permutation_eye(3)
        self.assertEqual(len(result), 3)
        self.assertEqual(len(result[0]), 3)

    def test_is_irreducible_matrix(self):
        matrix = [
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0]
        ]
        self.assertTrue(self.ifn.is_irreducible_matrix(matrix))

    def test_is_premagic_matrix(self):
        matrix = [
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0]
        ]
        self.assertTrue(self.ifn.is_premagic_matrix(matrix))

    def test_is_row_stochastic_matrix(self):
        matrix = [
            [0.5, 0.5, 0],
            [0, 0.5, 0.5],
            [0.5, 0, 0.5]
        ]
        self.assertTrue(self.ifn.is_row_stochastic_matrix(matrix))

    def test_is_ideal_flow_matrix(self):
        matrix = [
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0]
        ]
        self.assertTrue(self.ifn.is_ideal_flow_matrix(matrix))

    def test_adj_list_to_matrix(self):
        adjL = {
            'a': {'b': 1},
            'b': {'c': 1},
            'c': {'a': 1}
        }
        expected = [
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0]
        ]
        result = self.ifn.adj_list_to_matrix(adjL)
        self.assertTrue(np.array_equal(result, expected))

    def test_matrix_to_adj_list(self):
        matrix = [
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0]
        ]
        expected = {
            'a': {'b': 1},
            'b': {'c': 1},
            'c': {'a': 1}
        }
        result = self.ifn.matrix_to_adj_list(matrix)
        self.assertEqual(result, expected)

    def test_is_non_empty_adj_list(self):
        adjL = {
            'a': {'b': 1},
            'b': {'c': 1},
            'c': {'a': 1}
        }
        self.assertTrue(self.ifn.is_non_empty_adj_list(adjL))

    # def test_save_load_adj_list(self):
    #     adjL = {
    #         'a': {'b': 1},
    #         'b': {'c': 1},
    #         'c': {'a': 1}
    #     }
    #     self.ifn.save_adj_list(adjL, 'test_adj_list.json')
    #     loaded_adjL = self.ifn.load_adj_list('test_adj_list.json')
    #     self.assertEqual(adjL, loaded_adjL)
    def test_save_load_adj_list(self):
        adjL = {
            'a': {'b': 1},
            'b': {'c': 1},
            'c': {'a': 1}
        }

        # Use a temporary file for testing file save/load
        with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as tmp_file:
            tmp_filename = tmp_file.name

        try:
            # Save the adjacency list to a temporary file
            self.ifn.save_adj_list(adjL, tmp_filename)

            # Load the adjacency list from the temporary file
            loaded_adjL = self.ifn.load_adj_list(tmp_filename)

            # Compare the original and loaded adjacency lists
            self.assertEqual(adjL, loaded_adjL)
        
        finally:
            # Ensure the temporary file is removed after the test
            if os.path.exists(tmp_filename):
                os.remove(tmp_filename)

    def test_find_a_cycle(self):
        adjL = {
            'a': {'b': 1},
            'b': {'c': 1},
            'c': {'a': 1}
        }
        result = self.ifn.find_a_cycle('a', 'c', adjL)
        self.assertEqual(result, 'abc')

    def test_find_cycles(self):
        matrix = [
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0]
        ]
        expected =['abc']
        result = self.ifn.find_cycles(matrix)
        self.assertEqual(result, expected)

    def test_find_all_permutation_cycles(self):
        matrix = [
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0]
        ]
        expected = {'abc', 'bca', 'cab'}
        result = self.ifn.find_all_permutation_cycles(matrix)
        self.assertEqual(result, expected)
    
    def random_ifn(self, numNode=5):
        # generate random IFN 
        numLink=numNode * 2 - numNode + 1
        C=self.ifn.rand_irreducible(numNode,numLink)
        F=self.ifn.capacity_to_ideal_flow(C)
        scaling=self.ifn.global_scaling(F,'int')
        F=self.ifn.equivalent_ifn(F,scaling)
        return F

    def test_string_to_matrix(self):
        cycle_string = 'abcde+3abce'
        expected_matrix = np.array([
            [0, 4, 0, 0, 0],
            [0, 0, 4, 0, 0],
            [0, 0, 0, 1, 3],
            [0, 0, 0, 0, 1],
            [4, 0, 0, 0, 0]
        ])
        result = self.net.string_to_matrix(cycle_string)
        np.testing.assert_array_equal(result, expected_matrix)

    # def test_extract_first_k_terms(self):
    #     cycle_string = 'anckmblegjd+anckmbljd+anckmbgjd'
    #     expected = 'anckmblegjd + anckmbljd'
    #     result = self.net.extract_first_k_terms(cycle_string, 2)
    #     self.assertEqual(result, expected)

    # def test_extract_last_k_terms(self):
    #     cycle_string = 'anckmblegjd+anckmbljd+anckmbgjd'
    #     expected = 'anckmbljd + anckmbgjd'
    #     result = self.net.extract_last_k_terms(cycle_string, 2)
    #     self.assertEqual(result, expected)

    # def test_parse_terms_to_dict(self):
    #     cycle_string = 'anckmblegjd+anckmbljd+2anckmbgjd'
    #     expected_dict = {'anckmbgjd': 2, 'anckmblegjd': 1, 'anckmbljd': 1}
    #     result = self.net.parse_terms_to_dict(cycle_string)
    #     self.assertEqual(result, expected_dict)

    # def test_parse_terms_to_dict_combined(self):
    #     cycle_string = '2anckmblegjd+3anckmblegjd'
    #     expected_dict = {'anckmblegjd': 5}
    #     result = self.net.parse_terms_to_dict(cycle_string)
    #     self.assertEqual(result, expected_dict)

    # def test_generate_random_terms(self):
    #     cycle_dict = {'anckmblegjd': 1, 'anckmbljd': 1, 'anckmbgjd': 2}
    #     result = self.net.generate_random_terms(cycle_dict, 2, is_premier=True)
    #     self.assertTrue(all(term in cycle_dict.keys() for term in result.split(' + ')))

    # def test_generate_random_terms2(self):
    #     cycle_dict = {'anckmblegjd': 1, 'anckmbljd': 1, 'anckmbgjd': 2}
    #     result = self.net.generate_random_terms(cycle_dict, 2, is_premier=True)
    #     result_terms = result.split(' + ')
    #     self.assertTrue(all(term in cycle_dict.keys() for term in result_terms))
    #     self.assertTrue(all('+' not in term for term in result_terms))  # Ensure no coefficients in terms

    # def test_is_premier(self):
    #     cycle_string = 'abkoclhdjegf+2djegfh'
    #     self.assertFalse(self.net.is_premier_signature(cycle_string))
    #     cycle_string_2 = 'abkoclhdjegf+djegfh'
    #     self.assertTrue(self.net.is_premier(cycle_string_2))

    # def test_is_premier_ifn(self):
    #     cycle_string ='12ahobgdc + 3bgdilncho + 18ln + 2dikjg + 4bgdio + 0bgdcho + 12bgdlncho + 106ho + 2ahobgdikjfmelnc + ahobgdilnc + 0bgdikjfmelncho + 0ahobgdlnc + 36bgo + 36bgho'
    #     F = self.net.string_to_matrix(cycle_string)
    #     self.assertTrue(self.ifn.is_ideal_flow_matrix(F))

    def test_decompose_compose_ifn(self):
        # string cycles of IFN must not change after decomposition and composition
        numNode=random.randint(3, 10) 
        # decompose
        F0=self.random_ifn(numNode)
        cycle_string1 = self.net.solve_cycles(F0)
        # compose
        F1 = self.net.string_to_matrix(cycle_string1)
        cycle_string2 = self.net.solve_cycles(F1)
        self.assertEqual(cycle_string1,cycle_string2)

    def test_smaller_kappa(self):
        # kappa of smaller cycles must be smaller or equal  
        numNode=random.randint(3, 10) 
        # decompose
        F0=self.random_ifn(numNode)
        kappa1=self.ifn.kappa(F0)
        cycle_string1 = self.net.solve_cycles(F0)
        cycle_string2 = self.net.extract_first_k_terms(cycle_string1, 2)
        canonical_cycle_string = self.net.canonize_signature(cycle_string2)
        F1 = self.net.string_to_matrix(canonical_cycle_string)
        kappa2 = self.ifn.kappa(F1)
        # print('cycle_string1',cycle_string1)
        # print('cycle_string2',cycle_string2)
        # print('canonical_cycle_string',canonical_cycle_string)
        # print('kappa1:',kappa1,'kappa2:',kappa2)
        # print(F0)
        # print(F1)
        self.assertTrue(kappa1>=kappa2)

if __name__ == '__main__':
    # unittest.main()
    unittest.main(argv=[''], verbosity=2, exit=False)
