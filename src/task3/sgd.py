import os, sys
from pprint import pprint
import time
import matplotlib.pyplot as plt
sys.path.append(os.getcwd())

import numpy as np
from src.functions.activation.relu import relu
from src.functions.activation.sigmoid import sigmoid
from src.functions.loss.binary_cross_entropy_loss import binary_cross_entropy_loss
from src.functions.utils.numerical_differential import numerical_differential
from src.functions.utils.load_graph_from_datasets import load_graph_from_datasets
from src.functions.utils.train_test_split import train_test_split

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
        np.random.seed(12)
        self.theta['W'] = np.random.normal(0, 0.4, [self.fvdim, self.fvdim])
        np.random.seed(123)
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
        for i in range(0, self.step):
            new_features_vectors = self._aggregate(feature_vectors, graph, self.theta['W'])
            feature_vectors = new_features_vectors

        return self._readout(feature_vectors)

    def predict(self, graphs):
        hgs = np.array([self._calc_hg(graph) for graph in graphs])
        s = np.dot(hgs, self.theta['A']) + self.theta['b'][0]
        pprint(s)
        return np.array([1 if sigmoid(value) > 0.5 else 0 for value in s])

    def accuracy(self, graphs, labels):
        data_size = graphs.shape[0]
        predict = self.predict(graphs)
        pprint(predict)
        pprint(labels)
        cnt = 0
        for i in range(batch_size):
            if predict[i] == labels[i]:
                cnt += 1
        
        return float(cnt / data_size)

    def loss(self, graphs, labels):
        # fix for batch processing
        data_size = graphs.shape[0]
        hgs = np.array([self._calc_hg(graph) for graph in graphs])
        s = np.dot(hgs, self.theta['A']) + self.theta['b'][0]
        return np.sum([binary_cross_entropy_loss(labels[i], s[i]) for i in range(data_size)]) / data_size

    def numerical_gradient(self, graph, y):
        loss_W = lambda W: self.loss(graph, y)
        grads = {}
        grads['W'] = numerical_differential(loss_W, self.theta['W'], self.perturbation)
        grads['A'] = numerical_differential(loss_W, self.theta['A'], self.perturbation)
        grads['b'] = numerical_differential(loss_W, self.theta['b'], self.perturbation)
        return grads

if __name__ == '__main__':
    start = time.time()
    # hyper parameter
    fvdim = 8
    step = 2
    learning_rate = 0.0001
    perturbation = 0.001
    epoch = 50
    batch_size = 50

    number_of_data = 2000
    print("---- start loading datasets ----")
    graphs, labels = load_graph_from_datasets(number_of_data)
    graphs_train, graphs_test, labels_train, labels_test = train_test_split(graphs, labels, train_size=0.8)
    print("---- finish loading datasets ----")
    gnn = Gnn(fvdim, step, learning_rate, perturbation)

    # learning
    npa_graphs_train = np.array(graphs_train)
    npa_labels_train = np.array(labels_train)
    npa_graphs_test = np.array(graphs_test)
    npa_labels_test = np.array(labels_test)
    train_size = npa_graphs_train.shape[0]
    iteration = int(train_size / batch_size)
    loss_list_train = []
    loss_list_test = []
    accuracy_list_train = []
    accuracy_list_test = []

    for i in range(epoch):
        rand_index = np.random.permutation(np.arange(train_size)).reshape(-1, batch_size)
        for j in range(iteration):
            batch_graphs = npa_graphs_train[rand_index[j]]
            batch_labels = npa_labels_train[rand_index[j]]
            grad = gnn.numerical_gradient(batch_graphs, batch_labels)
            for key in gnn.theta.keys():
                gnn.theta[key] -= gnn.learning_rate * grad[key]

            # store
            loss_train = gnn.loss(batch_graphs, batch_labels)
            loss_test = gnn.loss(npa_graphs_test, npa_labels_test)
            accuracy_train = gnn.accuracy(batch_graphs, batch_labels)
            accuracy_test = gnn.accuracy(npa_graphs_test, npa_labels_test)
            loss_list_train.append(loss_train)
            loss_list_test.append(loss_test)
            accuracy_list_train.append(accuracy_train)
            accuracy_list_test.append(accuracy_test)

            pprint(loss_train)
            pprint(accuracy_train)
            pprint(loss_test)
            pprint(accuracy_test)


    # creating figure
    x = np.arange(len(loss_list_train))
    plt.plot(x, loss_list_train, label="train")
    plt.plot(x, loss_list_test, linestyle="dashed", label="test")
    plt.savefig("src/task3/loss.png")
    plt.plot(x, accuracy_list_train, label="train")
    plt.plot(x, accuracy_list_test, linestyle="dashed", label="test")
    plt.savefig("src/task3/accuracy.png")

    pprint(loss_list_train)
    pprint(accuracy_list_train)
    pprint(loss_list_test)
    pprint(accuracy_list_test)
    elapsed_time = time.time() - start
    print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")
