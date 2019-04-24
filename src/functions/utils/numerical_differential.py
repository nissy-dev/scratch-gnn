import numpy as np

def numerical_differential(f, x, p):
    """function for calculating numerical_gradient.

    Args:
        f (Variable): loss function
        x (Variable): hyper parameter
        p (Variable): perturbation (real number)
    
    Returns:
        output : 
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        # self.theta[key]を上書き
        x[idx] = float(tmp_val) + p
        fxh = f(x) # f(x+h)
        # self.theta[key]を上書き
        x[idx] = tmp_val
        fx = f(x) # f(x)
        grad[idx] = (fxh - fx) / p        
        x[idx] = tmp_val # 値を元に戻す
        it.iternext() 

    return grad