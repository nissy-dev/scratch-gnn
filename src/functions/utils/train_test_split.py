import numpy as np

def train_test_split(X, Y, train_size):
    """function for calculating numerical_differential.

    Args:
        X (Variable): graphs (list)
        Y (Variable): labels (list)
        train_size (Variable): ratio of the train data 
    
    Returns:
        X_train, X_test, Y_train, Y_test
    """
    data_length = len(X)
    index = int(data_length * train_size)
    return X[0:index], X[index:data_length], Y[0:index], Y[index:data_length]