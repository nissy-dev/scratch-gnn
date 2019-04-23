import numpy as np

def binary_cross_entropy_loss(y, s):
    """binary_cross_entropy_loss function.

    Args:
        y (Variable): Correct label (In this case, 0 or 1 is expected)
        s (Variable): Input variable (In this case, the real number is expected)
    
    Returns:
        output (Variable): Output variable is real number
    """

    # for overflow
    if (s > 500):
        return (1 - y) * s
    if (s < -500):
        return y * (-s)

    return y * np.log(1 + np.exp(-s)) + (1 - y) * np.log(1 + np.exp(s))