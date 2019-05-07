import os, sys
sys.path.append(os.getcwd())

import time
import numpy as np
import matplotlib.pyplot as plt
from src.task3.gnn import Gnn
from src.task3.updater import SGD
from src.functions.utils.load_graph_from_datasets import load_graph_from_datasets
from src.functions.utils.train_test_split import train_test_split

if __name__ == '__main__':
    start = time.time()
    # hyper parameter
    fvdim = 8
    step = 2
    learning_rate = 0.0001
    perturbation = 0.001
    epoch = 50
    batch_size = 100
    gnn = Gnn(fvdim, step, learning_rate, perturbation)
    sgd = SGD(learning_rate)

    number_of_data = 2000
    graphs, labels = load_graph_from_datasets(number_of_data)
    graphs_train, graphs_test, labels_train, labels_test = train_test_split(graphs, labels, train_size=0.8)
    print("---- finish loading datasets ----")

    # data
    npa_graphs_train = np.array(graphs_train)
    npa_labels_train = np.array(labels_train)
    npa_graphs_test = np.array(graphs_test)
    npa_labels_test = np.array(labels_test)

    # learning
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

            # update
            grads = gnn.numerical_gradient(batch_graphs, batch_labels)
            sgd.update(gnn.theta, grads)

            # store
            loss_train = gnn.loss(batch_graphs, batch_labels)
            loss_test = gnn.loss(npa_graphs_test, npa_labels_test)
            accuracy_train = gnn.accuracy(batch_graphs, batch_labels)
            accuracy_test = gnn.accuracy(npa_graphs_test, npa_labels_test)
            loss_list_train.append(loss_train)
            loss_list_test.append(loss_test)
            accuracy_list_train.append(accuracy_train)
            accuracy_list_test.append(accuracy_test)

        print("------- " + str(i+1) + " epoch --------")
        print("train loss, test loss | " + str(loss_train) + ", " + str(loss_test))
        print("train acc, test acc | " + str(accuracy_train) + ", " + str(accuracy_test))

    # creating figure
    fig = plt.figure()
    x = np.arange(epoch * iteration)
    ax1 = fig.add_subplot(211)
    ax1.set_title('loss')
    ax1.plot(x, loss_list_train, label="train")
    ax1.plot(x, loss_list_test, linestyle="dashed", label="test")
    ax1.legend()
    ax2 = fig.add_subplot(212)
    ax2.set_title('accuracy')
    ax2.plot(x, accuracy_list_train, label="train")
    ax2.plot(x, accuracy_list_test, linestyle="dashed", label="test")
    ax2.legend()
    plt.subplots_adjust(hspace=0.4)
    plt.savefig("src/task3/sgd.png")

    elapsed_time = time.time() - start
    print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")
