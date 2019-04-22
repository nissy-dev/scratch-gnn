import numpy as np

def relu(x):
    """relu function.

    Args:
        x (Variable): Input variable (In this case, the vector is expected)
    
    Returns:
        output (Variable): Output variable(>0) whose shape is same with `x` 
    """
    return np.maximum(0, x)