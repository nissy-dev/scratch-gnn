import os, sys
sys.path.append(os.getcwd())

import numpy as np
from src.functions.activation.relu import relu
from src.functions.activation.sigmoid import sigmoid
from src.functions.loss.binary_cross_entropy_loss import binary_cross_entropy_loss

# 12 dimension
test_gragh = np.array([
    [0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0],
    [1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0],
    [1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0],
    [0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0],
    [1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0],
    [1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1],
    [0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1],
    [0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0],
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
        # 学習率 (ハイパーパラメータ)
        self.learning_rate = 0.0001
        # ハイパーパラメータ (更新するもの)
        # W: fvdim × fvdim matrix, A: fvdim vector
        self.theta = {}
        np.random.seed(12)
        self.theta['W'] = np.random.normal(0, 0.4, [self.fvdim, self.fvdim])
        np.random.seed(123)
        self.theta['A'] = np.random.normal(0, 0.4, self.fvdim)
        self.theta['b'] = 0

    def _get_init_feature_vectors(self, fvdim, number_of_node):
        # 次元fvdim、最初の要素が１それ以外が０のベクトルをノード分生成 
        tmp_vector = np.zeros((fvdim * number_of_node))
        tmp_vector[::fvdim] = 1
        return tmp_vector.reshape(number_of_node, fvdim)
    
    def _aggregate(self, feature_vectors, graph, number_of_node, W):
        # 集約処理
        vector_list = [np.sum(feature_vectors[graph[i]==1], axis=0) for i in range(0, number_of_node)]
        a = np.array(vector_list)
        return relu(np.dot(W, a.T).T)

    def _readout(self, feature_vectors):
        return np.sum(feature_vectors, axis=0)

    def _calc_hg(self):
        # initialize
        number_of_node = self.graph.shape[0]
        feature_vectors = self._get_init_feature_vectors(self.fvdim, number_of_node)
        for i in range(0, self.step):
            new_features_vectors = self._aggregate(
                feature_vectors,
                self.graph,
                number_of_node,
                self.theta['W']
            )
            # 更新
            feature_vectors = new_features_vectors

        return self._readout(feature_vectors)

    def predict(self, s):
        p = sigmoid(s)
        return 1 if p > 0.5 else 0

    def calc_loss(self, y):
        hg = self._calc_hg()
        s = np.dot(self.theta['A'], hg) + self.theta['b']
        return binary_cross_entropy_loss(y, s)

    def gradient(self, y):
        return

if __name__ == '__main__':
    gnn = Gnn(test_gragh)
    # 適当なラベル y=0
    loss = gnn.calc_loss(y=0)
    print(loss)
