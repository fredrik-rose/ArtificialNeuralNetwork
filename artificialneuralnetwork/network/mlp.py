"""
Multilayer perceptron (MLP) network.
"""
import numpy as np


def sigmoid(z):
    """
    The sigmoid function.
    Could e.g. be used as an activation function in an artificial neural network.
    :param z: The input.
    :return: Sigmoid(z).
    """
    return 1 / (1 + np.exp(-z))


class NeuralNetwork():
    """
    A multilayer perceptron (MLP), a class of feedforward artificial neural network.
    """
    def __init__(self, layer_sizes, activation_function=sigmoid):
        """
        Initialize weights and biases to random values.
        :param layer_sizes: Tuple containing the number of neurons in each layer, including input,
                            hidden and output layers. Example with 2 hidden layer: (3,4,3,1).
        :param activation_function: The neuron activation function.
        """
        self._weights = [np.random.randn(size, input_size)
                         for input_size, size in zip(layer_sizes[:-1], layer_sizes[1:])]
        self._biases = [np.random.randn(size, 1)
                        for size in layer_sizes[1:]]
        self._activation_function = activation_function

    def feedforward(self, x):
        """
        Runs the network on a given input.
        :param x: Input, shall be a column vector of the same length as the input layer.
        :return: Network output, a column vector of the same length as the output layer.
        """
        for weights, biases in zip(self._weights, self._biases):
            z = np.dot(weights, x) + biases
            x = self._activation_function(z)
        return x
