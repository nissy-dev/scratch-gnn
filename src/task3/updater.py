import numpy as np

class SGD:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        
    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.learning_rate * grads[key]


class MSGD:
    def __init__(self, learning_rate, momentum):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.w = None
        
    def update(self, params, grads):
        if self.w is None:
            self.w = {}
            for key, val in params.items():    
                self.w[key] = np.zeros_like(val)
                
        for key in params.keys():
            self.w[key] = - self.learning_rate * grads[key] + self.momentum * self.w[key]
            params[key] += self.w[key]
