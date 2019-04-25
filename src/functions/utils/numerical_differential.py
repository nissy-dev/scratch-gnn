import numpy as np

def numerical_differential(f, x, p):
    """function for calculating numerical_differential.

    Args:
        f (Variable): loss function
        x (Variable): hyper parameter
        p (Variable): perturbation (real number)
    
    Returns:
        output :  Output variable whose shape is same with `x`
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        # overwrite self.theta[key]
        x[idx] = float(tmp_val) + p
        fxh = f(x) # f(x+h)
        # overwrite self.theta[key]
        x[idx] = tmp_val
        fx = f(x) # f(x)
        grad[idx] = (fxh - fx) / p
        # restore self.theta[key]
        x[idx] = tmp_val
        it.iternext() 

    return grad