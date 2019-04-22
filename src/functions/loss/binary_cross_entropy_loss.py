import numpy as np

def binary_cross_entropy_loss(y, s):
    """sigmoid function.

    Args:
        y (Variable): Predicted value (In this case, 0 or 1 is expected)
        s (Variable): Input variable (In this case, the real number is expected)
    
    Returns:
        output (Variable): Output variable is real number
    """
  
    return y * np.log(1 + np.exp(-s)) + (1 - y) * np.log(1 + np.exp(s))