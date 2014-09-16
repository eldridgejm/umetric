import unittest
import umetric

import networkx as nx
import numpy as np

class TestCore(unittest.TestCase):

    def setUp(self):
        self.dendrogram = nx.DiGraph()

        self.dendrogram.add_nodes_from([0,1,2,3,4], distance=0.0)
        self.dendrogram.add_nodes_from([5,6,7,8])
        self.dendrogram.node[5]['distance'] = 3.0
        self.dendrogram.node[6]['distance'] = 1.0
        self.dendrogram.node[7]['distance'] = 5.0
        self.dendrogram.node[8]['distance'] = 10.0

        self.dendrogram.add_edges_from([(8,5), (8,7), (5,0), (5,6), (6,1), 
                                        (6,2), (7,3), (7,4)])

        self.ultrametric = np.array([
                                [0, 3, 3, 10, 10],
                                [3, 0, 1, 10, 10],
                                [3, 1, 0, 10, 10],
                                [10, 10, 10, 0, 5],
                                [10, 10, 10, 5, 0]
                                ], dtype=float)
    
    def test_dendrogram_to_ultrametric(self):
        ultrametric = umetric.dendrogram_to_ultrametric(self.dendrogram)
        np.testing.assert_allclose(ultrametric, self.ultrametric)

    def test_ultrametric_to_dendrogram(self):
        dendrogram = umetric.ultrametric_to_dendrogram(self.ultrametric)
        ultrametric = umetric.dendrogram_to_ultrametric(dendrogram)
        np.testing.assert_allclose(ultrametric, self.ultrametric)

    def test_is_ultrametric(self):
        not_an_ultrametric = self.ultrametric.copy()
        not_an_ultrametric[1,0] = 12

        self.assertTrue(umetric.is_ultrametric(self.ultrametric))
        self.assertFalse(umetric.is_ultrametric(not_an_ultrametric))


class TestFit(unittest.TestCase):

    def setUp(self):
        self.ultrametric = np.array([
                                [0, 3, 3, 10, 10],
                                [3, 0, 1, 10, 10],
                                [3, 1, 0, 10, 10],
                                [10, 10, 10, 0, 5],
                                [10, 10, 10, 5, 0]
                                ], dtype=float)
    
    def test_closest_l_infinity(self):
        ultrametric = umetric.fit.closest_l_infinity(self.ultrametric)
        np.testing.assert_allclose(ultrametric, self.ultrametric)

        metric = np.array([[0,1,2,3],
                           [0,0,1,2],
                           [0,0,0,1],
                           [0,0,0,0]], dtype=float)

        metric = metric + metric.T

        actual = np.array([[0,2,2,2],
                           [0,0,2,2],
                           [0,0,0,2],
                           [0,0,0,0]], dtype=float)

        actual = actual + actual.T

        ultrametric = umetric.fit.closest_l_infinity(metric)

        np.testing.assert_allclose(ultrametric, actual)



if __name__ == "__main__":
    unittest.main()
