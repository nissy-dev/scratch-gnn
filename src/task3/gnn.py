import numpy as np
from src.functions.activation.relu import relu
from src.functions.activation.sigmoid import sigmoid
from src.functions.loss.binary_cross_entropy_loss import binary_cross_entropy_loss
from src.functions.utils.numerical_differential import numerical_differential

class Gnn:
    def __init__(self, fvdim, step, learning_rate, perturbation):
        #  特徴ベクトルの次元 (ハイパーパラメータ)
        self.fvdim = fvdim
        # ステップ数 (ハイパーパラメータ)
        self.step = step
        # 学習率 (ハイパーパラメータ)
        self.learning_rate = learning_rate
        # 数値微分の摂動 (ハイパーパラメータ)
        self.perturbation = perturbation
        # ハイパーパラメータ (更新するもの)
        # W: fvdim × fvdim matrix, A: fvdim vector
        self.theta = {}
        # np.random.seed(1)
        self.theta['W'] = np.random.normal(0, 0.4, [self.fvdim, self.fvdim])
        # np.random.seed(12)
        self.theta['A'] = np.random.normal(0, 0.4, self.fvdim)
        self.theta['b'] = np.array([0], dtype=float)

    def _get_init_feature_vectors(self, fvdim, graph):
        number_of_node = graph.shape[0]
        tmp_vector = np.zeros((fvdim * number_of_node))
        tmp_vector[::fvdim] = 1
        return tmp_vector.reshape(number_of_node, fvdim)
    
    def _aggregate(self, feature_vectors, graph, W):
        number_of_node = graph.shape[0]
        vector_list = [np.sum(feature_vectors[graph[i]==1], axis=0) for i in range(number_of_node)]
        a = np.array(vector_list)
        return relu(np.dot(W, a.T).T)

    def _readout(self, feature_vectors):
        return np.sum(feature_vectors, axis=0)

    def _calc_hg(self, graph):
        feature_vectors = self._get_init_feature_vectors(self.fvdim, graph)
        for i in range(self.step):
            new_features_vectors = self._aggregate(feature_vectors, graph, self.theta['W'])
            feature_vectors = new_features_vectors

        return self._readout(feature_vectors)

    def predict(self, graphs):
        hgs = np.array([self._calc_hg(graph) for graph in graphs])
        s = np.dot(self.theta['A'], hgs.T).T + self.theta['b'][0]
        return np.array([1 if sigmoid(value) > 0.5 else 0 for value in s])

    def accuracy(self, graphs, labels):
        data_size = graphs.shape[0]
        predict = self.predict(graphs)
        cnt = 0
        for i in range(data_size):
            if predict[i] == labels[i]:
                cnt += 1
        
        return float(cnt / data_size)

    def loss(self, graphs, labels):
        # fix for batch processing
        data_size = graphs.shape[0]
        hgs = np.array([self._calc_hg(graph) for graph in graphs])
        s = np.dot(self.theta['A'], hgs.T).T + self.theta['b'][0]
        return np.sum([binary_cross_entropy_loss(labels[i], s[i]) for i in range(data_size)]) / data_size

    def numerical_gradient(self, graphs, labels):
        loss_W = lambda W: self.loss(graphs, labels)
        grads = {}
        grads['W'] = numerical_differential(loss_W, self.theta['W'], self.perturbation)
        grads['A'] = numerical_differential(loss_W, self.theta['A'], self.perturbation)
        grads['b'] = numerical_differential(loss_W, self.theta['b'], self.perturbation)
        return grads
