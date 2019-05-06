import os, sys
sys.path.append(os.getcwd())

import unittest
import numpy as np
from numpy.testing import assert_array_equal
from src.task1.gnn import Gnn

# 問題文にあるグラフ
gragh = np.array([
    [0, 1, 0, 0],
    [1, 0, 1, 1],
    [0, 1, 0, 1],
    [0, 1, 1, 0],
])
gnn = Gnn(fvdim=3, step=2)

class TestGnn(unittest.TestCase):
    """test class of gnn.py
    """
    def test_aggregate_1(self):
        """test method for aggregate_1
        """
        feature_vectors = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]])
        result = gnn.aggregate_1(feature_vectors, gragh, gragh.shape[0])
        expected = np.array([[1, 0, 0], [3, 0, 0], [2, 0, 0], [2, 0, 0]])
        assert_array_equal(result, expected)

    def test_aggregate_2(self):
        """test method for aggregate_2
        """
        W = np.array([[1, 0, 0], [-1, 0, 0], [1, 0, 0]])
        a = np.array([[1, 0, 0], [3, 0, 0], [2, 0, 0], [2, 0, 0]])
        result = gnn.aggregate_2(W, a)
        expected = np.array([[1, 0, 1], [3, 0, 3], [2, 0, 2], [2, 0, 2]])
        assert_array_equal(result, expected)

    def test_readout(self):
        """test method for readout
        """
        v = np.array([[1, 0, 1], [3, 0, 3], [2, 0, 2], [2, 0, 2]])
        result = gnn.readout(v)
        expected = np.array([8, 0, 8])
        assert_array_equal(result, expected)

    def test_calc_hg(self):
        """test method for calc_hg
        """
        W = np.array([[1, 0, 0], [-1, 0, 0], [1, 0, 0]])
        gnn.W = W
        result = gnn.calc_hg(gragh)
        expected = expected = np.array([18, 0, 18])
        assert_array_equal(result, expected)

if __name__ == "__main__":
    unittest.main()
