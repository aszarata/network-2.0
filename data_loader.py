import numpy as np

class DataLoader:

    def __init__(self, X, y, size):
        self.X = X
        self.y = y
        self.size = size

    def get_batch(self):
        """
        Generates a mini-batch from the training data.
        
        Args:
            X: Training input data.
            y: Training target data.
            size: Size of the mini-batch.
        
        Returns:
            A mini-batch of input and target data.
        """

        start_idx = np.random.randint(0, len(self.X) - self.size + 1)
        mini_batch_X = self.X[start_idx:start_idx + self.size]
        mini_batch_y = self.y[start_idx:start_idx + self.size]
        return mini_batch_X, mini_batch_y