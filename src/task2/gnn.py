import os, sys
import matplotlib.pyplot as plt
sys.path.append(os.getcwd())

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
        np.random.seed(1)
        self.theta['W'] = np.random.normal(0, 0.4, [self.fvdim, self.fvdim])
        np.random.seed(12)
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

    def predict(self, graph):
        hg = self._calc_hg(graph)
        s = np.dot(self.theta['A'], hg) + self.theta['b'][0]
        return 1 if sigmoid(s) > 0.5 else 0

    def loss(self, graph, y):
        hg = self._calc_hg(graph)
        s = np.dot(self.theta['A'], hg) + self.theta['b'][0]
        return binary_cross_entropy_loss(y, s)

    def numerical_gradient(self, graph, y):
        loss_W = lambda W: self.loss(graph, y)
        grads = {}
        grads['W'] = numerical_differential(loss_W, self.theta['W'], self.perturbation)
        grads['A'] = numerical_differential(loss_W, self.theta['A'], self.perturbation)
        grads['b'] = numerical_differential(loss_W, self.theta['b'], self.perturbation)
        return grads

if __name__ == '__main__':
    # data
    graph = np.array([
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0],
    ])
    label = 1

    # hyper parameter
    fvdim = 8
    step = 2
    learning_rate = 0.0001
    perturbation = 0.001
    epoch = 100
    gnn = Gnn(fvdim, step, learning_rate, perturbation)

    train_loss_list = []
    # learning
    for i in range(epoch):
        # calculate gradient
        grad = gnn.numerical_gradient(graph, label)
        # update hyper parameter
        for key in gnn.theta.keys():
            gnn.theta[key] -= gnn.learning_rate * grad[key]

        loss = gnn.loss(graph, label)
        train_loss_list.append(loss)

    print("final loss value: {}".format(train_loss_list[-1]))

    # creating figure
    x = np.arange(epoch)
    plt.plot(x, train_loss_list, label="loss")
    plt.savefig("src/task2/loss.png")
