import numpy as np

class Adam:
    def __init__(self, learning_rate=0.001 ,beta1=0.9, beta2=0.99):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.m = None
        self.v = None
        self.t = 0
        
    def update(self, params, grads):
        if self.m is None:
            self.m = {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)

        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

        self.t += 1
        for key in params.keys():
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grads[key] ** 2)
            mhat = self.m[key] / (1 - self.beta1 ** self.t)
            vhat = self.v[key] / (1 - self.beta2 ** self.t)
            params[key] -= self.learning_rate * mhat / (np.sqrt(vhat) + 1e-8)
