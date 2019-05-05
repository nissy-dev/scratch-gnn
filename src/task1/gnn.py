import os, sys
sys.path.append(os.getcwd())

import numpy as np
from src.functions.activation.relu import relu

class Gnn:
    def __init__(self, fvdim, step):
        #  特徴ベクトルの次元 (ハイパーパラメータ)
        self.fvdim = fvdim
        # ステップ数 (ハイパーパラメータ)
        self.step = step
        # ハイパーパラメーター fvdim × fvdim matrix
        np.random.seed(1)
        self.W = np.random.normal(0, 0.4, [self.fvdim, self.fvdim])

    def get_init_feature_vectors(self, fvdim, number_of_node):
        # 次元fvdim、最初の要素が１それ以外が０のベクトルをノード分生成 
        tmp_vector = np.zeros((fvdim * number_of_node))
        tmp_vector[::fvdim] = 1
        return tmp_vector.reshape(number_of_node, fvdim)

    def aggregate_1(self, feature_vectors, graph, number_of_node):
        # 集約処理1
        vector_list = []
        for i in range(number_of_node):
            tmp_vector = np.sum(feature_vectors[graph[i]==1], axis=0)
            vector_list.append(tmp_vector)
        return np.array(vector_list)
    
    def aggregate_2(self, W, a):
        # 集約処理2
        new_feature_vectors = relu(np.dot(W, a.T).T)
        return new_feature_vectors

    def readout(self, feature_vectors):
        return np.sum(feature_vectors, axis=0)

    def calc_hg(self, graph):
        number_of_node = graph.shape[0]
        feature_vectors = self.get_init_feature_vectors(self.fvdim, number_of_node)
        for i in range(self.step):
            tmp_vectors = self.aggregate_1(feature_vectors, graph, number_of_node)
            feature_vectors = self.aggregate_2(self.W, tmp_vectors)

        return self.readout(feature_vectors)
