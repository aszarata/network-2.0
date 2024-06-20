import pickle

class Model:

    def __init__(self, learning_rate=0.1):
        """
        Initializes the network with an empty list of layers and sets the 
        loss function to mean squared error (MSE).
        """
        self.layers = []
        self.learning_rate = learning_rate

        self.is_training = True # State of trainig or state of evaluation

    def set_learning_rate(self, lr):
        """
        Sets a new learning rate for the model.
        """
        self.learning_rate = lr
    
    def add_layer(self, layer):
        """
        Adds a layer to the neural network.
        
        Args:
            layer: An instance of the Layer class to be added to the network.
        """
        self.layers.append(layer)


    def forward(self, X_batch):
        """
        Performs a single epoch of evaluating the X_batch.

        Args:
            X_batch: A numpy array of input data.

        Return:
            The output evaluation of the model.
        """

        inputs = X_batch

        for layer in self.layers:
            inputs = layer.train_forward(inputs)
        
        return inputs
    
    def predict(self, X):
        """
        Makes predictions using the trained neural network.
        
        Args:
            X: Input data for making predictions.
        
        Returns:
            Predictions made by the network.
        """
        y_pred = X
        for layer in self.layers:
            y_pred = layer.forward(y_pred)
        return y_pred
    

    def backward(self, y_batch):
        """
        Performs a single epoch of updating the weights and errors using back propagation based on true value.

        Args:
            y_batch: A numpy array of real outputs.
        """
        outputs = y_batch
        output_layer = self.layers[-1]

        delta_tmp = output_layer.calculate_output_delta(outputs)

        if self.is_training:
            output_layer.update_weights(lr=self.learning_rate)
        
        for layer in reversed(self.layers[:-1]):
            delta_tmp = layer.calculate_delta(delta_tmp)

            if self.is_training:
                layer.update_weights(lr=self.learning_rate)

        return delta_tmp
    
    def backward_from(self, delta):
        """
        Performs a single epoch of updating the weights and errors using back propagation continuing from last delta.

        Args:
            delta: A numpy array of real outputs.
        """
        delta_tmp = delta
        for layer in reversed(self.layers):
            delta_tmp = layer.calculate_delta(delta_tmp)

            if self.is_training:
                layer.update_weights(lr=self.learning_rate)

        return delta_tmp


    def eval(self):
        """
        Sets the model into evaluation state.
        """
        self.is_training=False

    def train(self):
        """
        Sets the model into training state.
        """
        self.is_training=True


    def save(self, filename):
        """
        Saves the trained model to a file.
        
        Args:
            filename: Path to the file where the model will be saved.
        """
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, filename):
        """
        Loads a trained model from a file.
        
        Args:
            filename: Path to the file from which the model will be loaded.
        
        Returns:
            The loaded model.
        """
        with open(filename, 'rb') as file:
            loaded_object = pickle.load(file)
        return loaded_object