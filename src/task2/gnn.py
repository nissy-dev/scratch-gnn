import os, sys
sys.path.append(os.getcwd())

import numpy as np
from src.functions.activation.relu import relu
from src.functions.activation.sigmoid import sigmoid
from src.functions.loss.binary_cross_entropy_loss import binary_cross_entropy_loss
from src.functions.utils.numerical_differential import numerical_differential

class Gnn:
    def __init__(self, fvdim, step, learning_rate, perturbation, epoch):
        #  特徴ベクトルの次元 (ハイパーパラメータ)
        self.fvdim = fvdim
        # ステップ数 (ハイパーパラメータ)
        self.step = step
        # 学習率 (ハイパーパラメータ)
        self.learning_rate = learning_rate
        # 数値微分の摂動 (ハイパーパラメータ)
        self.perturbation = perturbation
        # 学習の際のエポック数 (ハイパーパラメーター)
        self.epoch = epoch
        # ハイパーパラメータ (更新するもの)
        # W: fvdim × fvdim matrix, A: fvdim vector
        self.theta = {}
        np.random.seed(12)
        self.theta['W'] = np.random.normal(0, 0.4, [self.fvdim, self.fvdim])
        np.random.seed(123)
        self.theta['A'] = np.random.normal(0, 0.4, self.fvdim)
        self.theta['b'] = np.array([0], dtype=float)

    def _get_init_feature_vectors(self, fvdim, graph):
        # 次元fvdim、最初の要素が１それ以外が０のベクトルをノード分生成 
        number_of_node = graph.shape[0]
        tmp_vector = np.zeros((fvdim * number_of_node))
        tmp_vector[::fvdim] = 1
        return tmp_vector.reshape(number_of_node, fvdim)
    
    def _aggregate(self, feature_vectors, graph, W):
        # 集約処理
        number_of_node = graph.shape[0]
        vector_list = [np.sum(feature_vectors[graph[i]==1], axis=0) for i in range(0, number_of_node)]
        a = np.array(vector_list)
        return relu(np.dot(W, a.T).T)

    def _readout(self, feature_vectors):
        return np.sum(feature_vectors, axis=0)

    def calc_hg(self, graph):
        feature_vectors = self._get_init_feature_vectors(self.fvdim, graph)
        for i in range(0, self.step):
            new_features_vectors = self._aggregate(feature_vectors, graph, self.theta['W'])
            feature_vectors = new_features_vectors

        return self._readout(feature_vectors)

    def predict(self, s):
        p = sigmoid(s)
        return 1 if p > 0.5 else 0

    def loss(self, graph, y):
        hg = self.calc_hg(graph)
        s = np.dot(self.theta['A'], hg) + self.theta['b'][0]
        return binary_cross_entropy_loss(y, s)

    def numerical_gradient(self, graph, y):
        loss_W = lambda W: self.loss(graph, y)
        grads = {}
        grads['W'] = numerical_differential(loss_W, self.theta['W'], self.perturbation)
        grads['A'] = numerical_differential(loss_W, self.theta['A'], self.perturbation)
        grads['b'] = numerical_differential(loss_W, self.theta['b'], self.perturbation)
        return grads

    def fit(self, graph, y):
        train_loss_list = []
        for i in range(self.epoch):
            # 勾配の計算
            grad = self.numerical_gradient(graph, y)
            # パラメータの更新
            for key in self.theta.keys():
                self.theta[key] -= self.learning_rate * grad[key]
            
            # 学習経過の記録
            loss = self.loss(graph, y)
            train_loss_list.append(loss)

        return train_loss_list

if __name__ == '__main__':
    # 12 dimension
    graph = np.array([
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
    gnn = Gnn(fvdim=8, step=2, learning_rate=0.0001, perturbation=0.001, epoch=10)
    loss_list = gnn.fit(graph=graph, y=0)
    print(loss_list)
