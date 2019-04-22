import os, sys
sys.path.append(os.getcwd())

import numpy as np
from src.functions.activation.relu import relu
from src.functions.activation.sigmoid import sigmoid
from src.functions.loss.binary_cross_entropy_loss import binary_cross_entropy_loss

# 7 dimension
test_gragh = np.array([
    [0, 1, 1, 0, 0, 0, 0],
    [1, 0, 1, 1, 0, 0, 0],
    [1, 1, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 1, 1, 0],
    [0, 0, 1, 1, 0, 0, 1],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0]
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
        np.random.seed(1234)
        self.W = np.random.normal(0, 0.4, [self.fvdim, self.fvdim])
        np.random.seed(12345)
        # ハイパーパラメーター fvdim vector
        self.A = np.random.normal(0, 0.4, self.fvdim)
        # ハイパーパラメーター
        self.b = 0
        # ノード数
        self._number_of_node = self.graph.shape[0]

    def _get_init_feature_vectors(self, fvdim, number_of_node):
        # 次元fvdim、最初の要素が１それ以外が０のベクトルをノード分生成 
        tmp_vector = np.zeros((fvdim * number_of_node))
        tmp_vector[::fvdim] = 1
        return tmp_vector.reshape(number_of_node, fvdim)
    
    def _aggregate(self, feature_vectors, graph, number_of_node, W):
        # 集約処理
        vector_list = [np.sum(feature_vectors[graph[i]==1], axis=0) for i in range(0, number_of_node)]
        a = np.array(vector_list)
        new_feature_vectors = relu(np.dot(W, a.T).T)
        return new_feature_vectors

    def _readout(self, feature_vectors):
        return np.sum(feature_vectors, axis=0)

    def _calc_hg(self):
        # initialize
        feature_vectors = self._get_init_feature_vectors(self.fvdim, self._number_of_node)
        for i in range(0, self.step):
            tmp_vectors = self._aggregate(
                feature_vectors,
                self.graph,
                self._number_of_node,
                self.W
            )
            # 更新
            feature_vectors = tmp_vectors

        return self._readout(feature_vectors)

    def _get_weighted_sum_of_hg(self, hg, a, b):
        return np.dot(hg, a) + b

    def _predict(self, s):
        probability = sigmoid(s)
        return 1 if probability > 0.5 else 0

    def calc_loss(self):
        hg = self._calc_hg()
        s = self._get_weighted_sum_of_hg(hg, self.A, self.b)
        y = self._predict(s)
        return binary_cross_entropy_loss(y, s)


if __name__ == '__main__':
    gnn = Gnn(test_gragh)
    loss = gnn.calc_loss()
    print(loss)
