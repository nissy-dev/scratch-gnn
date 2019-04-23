import os, sys
sys.path.append(os.getcwd())

import numpy as np
from src.functions.activation.relu import relu

# 4 dimension
test_gragh = np.array([
    [0, 1, 0, 0],
    [1, 0, 1, 1],
    [0, 1, 0, 1],
    [0, 1, 1, 0],
])

class Gnn:
    def __init__(self, graph):
        """
        Parameters
        ----------
        graph : array
            グラフ(隣接行列)
        """
        self.graph = graph
        #  特徴ベクトルの次元 (ハイパーパラメータ)
        self.fvdim = 8
        # ステップ数 (ハイパーパラメータ)
        self.step = 2
        # ハイパーパラメーター fvdim × fvdim matrix
        np.random.seed(1)
        self.W = np.random.normal(0, 0.4, [self.fvdim, self.fvdim])

    def _get_init_feature_vectors(self, fvdim, number_of_node):
        # 次元fvdim、最初の要素が１それ以外が０のベクトルをノード分生成 
        tmp_vector = np.zeros((fvdim * number_of_node))
        tmp_vector[::fvdim] = 1
        return tmp_vector.reshape(number_of_node, fvdim)

    def _aggregate_1(self, feature_vectors, graph, number_of_node):
        # 集約処理1
        vector_list = []
        for i in range(0, number_of_node):
            tmp_vector = np.sum(feature_vectors[graph[i]==1], axis=0)
            vector_list.append(tmp_vector)
        return np.array(vector_list)
    
    def _aggregate_2(self, W, a):
        # 集約処理2
        new_feature_vectors = relu(np.dot(W, a.T).T)
        return new_feature_vectors

    def _readout(self, feature_vectors):
        return np.sum(feature_vectors, axis=0)

    def calc_hg(self):
        # initialize
        number_of_node = self.graph.shape[0]
        feature_vectors = self._get_init_feature_vectors(self.fvdim, number_of_node)
        for i in range(0, self.step):
            tmp_vectors = self._aggregate_1(feature_vectors, self.graph, number_of_node)
            feature_vectors = self._aggregate_2(self.W, tmp_vectors)

        return self._readout(feature_vectors)

if __name__ == '__main__':
    gnn = Gnn(graph=test_gragh)
    hg = gnn.calc_hg()
    print(hg)

# testはmethodごとのtest + 簡単なグラフでのテストを書く