import numpy as np

def load_graph_from_datasets(num):
    """function for loading datasets.

    Args:
        num (Variable): number of data which you want
    
    Returns:
        graphs, labels : list for graph , list for label
    """

    ids = range(num)
    graphs = []
    labels = []
    for id in ids:
        # loading graph
        graph_path = 'src/datasets/train/{}_graph.txt'.format(id)
        with open(graph_path) as f:
            graph = []
            for s in f.readlines():
                row = s.strip().split()
                int_row = [int(val) for val in row]
                if len(int_row) > 1:
                    graph.append(int_row)

        graphs.append(np.array(graph))

        # loading label
        label_path = 'src/datasets/train/{}_label.txt'.format(id)
        with open(label_path) as f:
            for s in f.readlines():
                label = int(s.strip())

        labels.append(label)

    return graphs, labels