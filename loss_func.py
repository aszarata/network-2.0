import numpy as np

def MSE(y_pred, y_true):
    loss = np.mean((y_pred.reshape(1, -1) - y_true.reshape(1, -1))**2)
    return loss