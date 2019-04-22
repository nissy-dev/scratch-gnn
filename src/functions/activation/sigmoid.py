import numpy as np

def sigmoid(x):
    """sigmoid function.

    Args:
        x (Variable): Input variable (In this case, the real number is expected)
    
    Returns:
        output (Variable): Output variable is real number (0~1)
    """
    return 1.0 / (1.0 + np.exp(-x))