import numpy as np

def crgraph(dim):
    """function for creating random graph.

    Args:
        dim (Variable): dimension of output matrix
    
    Returns:
        output : symmetric matrix
    """
    A = np.zeros((dim,dim))
    for i in range(dim):
        for j in range(i):
            value = np.random.choice([0, 1])
            A[j][i] = value
            A[i][j] = value

    return A