import numpy as np

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
    def __init__(self, graph, ndim=4, step=5):
        """
        Parameters
        ----------
        graph : array
            グラフ(隣接行列)
        ndim: int
            特徴ベクトルの次元 (ハイパーパラメータ)
        step: int
            ステップ数 (ハイパーパラメータ)
        """
        self.graph = graph
        self.ndim = ndim
        self.step = step
        self._W = self._get_W(ndim)
        # ノード数
        self._number_of_node = self.graph.shape[0]

    def _get_W(self, ndim):
        # これもハイパーパラメーター (いずれは外から受け取る)
        np.random.seed(1234)
        return np.random.randn(ndim, ndim)

    def _get_init_feature_vectors(self, ndim, number_of_node):
        # 次元ndim、最初の要素が１それ以外が０のベクトルをノード分生成 
        tmp_vector = np.zeros((ndim * number_of_node))
        tmp_vector[::ndim] = 1
        return tmp_vector.reshape(number_of_node, ndim)

    def _aggregate_1(self, feature_vectors, graph, number_of_node):
        # 集約処理1
        vector_list = []
        for i in range(0, number_of_node):
            tmp_vector = np.sum(feature_vectors[graph[i]==1], axis=0)
            vector_list.append(tmp_vector)
      # 1 line
      # vector_list = [np.sum(feature_vectors[graph[i]==1], axis=0) for i in range(0, number_of_node)]
        return np.array(vector_list)
    
    def _aggregate_2(self, W, a):
        # 集約処理2
        new_feature_vectors = self._relu(np.dot(W, a.T).T)
        return new_feature_vectors

    def _relu(self, x):
        return np.maximum(0, x)

    def _readout(self, feature_vectors):
        return np.sum(feature_vectors, axis=0)

    def get_hg(self):
        # initialize
        feature_vectors = self._get_init_feature_vectors(self.ndim, self._number_of_node)
        for i in range(0, self.step):
            tmp_vectors = self._aggregate_1(feature_vectors, self.graph, self._number_of_node)
            feature_vectors = self._aggregate_2(self._W, tmp_vectors)

        return self._readout(feature_vectors)

if __name__ == '__main__':
    gnn = Gnn(test_gragh)
    hg = gnn.get_hg()
    print(hg)