import numpy as np
from scipy.special import expit

# Hard threshold
class HardThreshold:
    def fn(self, x):
        return np.where(x > 0, 1, 0)

    def drv(self, x):
        return np.zeros_like(x)    

# Logistic
class Logistic:
    def fn(self, x):
        return expit(x)

    def drv(self, x):
        return self.fn(x) * (1 - self.fn(x))

# Identity
class Identity:
    def fn(self, x):
        return x

    def drv(self, x):
        return np.ones_like(x)


# ReLU
class ReLU:
    def fn(self, x):
        return np.maximum(0, x)

    def drv(self, x):
        return np.where(x > 0, 1, 0)
    

# Leaky ReLU
class LeakyReLU:
    def __init__(self, ratio):
        self.ratio = ratio

    def fn(self, x):
        return np.maximum(0, x) + np.minimum(0, x) * self.ratio    
    
    def drv(self, x):
        return np.where(x > 0, 1, self.ratio)


# Softmax
class Softmax:
    def fn(self, x):
            exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
            return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def drv(self, x):
        s = self.fn(x)
        return s * (1 - s)