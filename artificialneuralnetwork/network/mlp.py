"""
Multilayer perceptron (MLP) network.
"""
import random as rnd

import numpy as np


def sigmoid(z):
    """
    The sigmoid function.
    Could e.g. be used as an activation function in an artificial neural network.
    :param z: The input.
    :return: Sigmoid(z).
    """
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(z):
    """
    The derivative of the sigmoid function.
    :param z: The input.
    :return: Sigmoid'(z).
    """
    return np.exp(-z) / ((1 + np.exp(-z)) ** 2)


class NeuralNetwork():
    """
    A multilayer perceptron (MLP), a class of feedforward artificial neural network.
    The cost function C used is defined as
      C = 1/n * S(1/2 * ||a - y||^2),
    where
      a is the actual output from the network,
      y is the expected output from the network,
      n is the number of training samples,
      S is the sum over all traing samples,
      ||v|| denotes the length of vector v.
    """
    def __init__(self, layer_sizes, activation_function=(sigmoid, sigmoid_derivative)):
        """
        Initialize weights and biases to random values.
        :param layer_sizes: Tuple containing the number of neurons in each layer, including input,
                            hidden and output layers. Example with 2 hidden layer: (3,4,3,1).
        :param activation_function: Tuple containing the neuron activation function and its derivative.
        """
        self._number_of_layers = len(layer_sizes)
        self._biases = np.asarray([np.random.randn(size, 1)
                                   for size in layer_sizes[1:]])
        self._weights = np.asarray([np.random.randn(size, input_size)
                                    for input_size, size in zip(layer_sizes[:-1], layer_sizes[1:])])
        self._activation_function = activation_function[0]
        self._activation_function_derivative = activation_function[1]


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


    def train(self, training_data, epochs, batch_size, learning_rate):
        """
        Trains the network using stochastic gradient descent.
        :param training_data: List of training pairs (input, expected output), which must match the
                              size of the input and output layers, respectively.
        :param epochs: Number of training epochs, i.e. number of passes over all the training samples.
        :param batch_size: Number of training samples in a batch, used to estimate the gradient for
                           a single gradient descent step.
        :param learning_rate: The gradient descent step size.
        """
        for _ in range(epochs):
            rnd.shuffle(training_data)
            for i in range(0, len(training_data), batch_size):
                batch = training_data[i:i + batch_size]
                bias_gradients, weight_gradients = self._gradient(batch)
                self._biases -= learning_rate * bias_gradients
                self._weights -= learning_rate * weight_gradients


    def _gradient(self, training_data):
        """
        Calculates the gradient of the cost function for all training samples.
        Use the fact that the derivative of a sum is equal to the sum of the derivatives of each
        term, when taking the derivative of the cost function.
        :param training_data: List of training pairs (input, expected output), which must match the
                              size of the input and output layers, respectively.
        :return: Tuple containing bias gradients and weight gradients.
        """
        # TODO: It is possible to change the backpropagation function slightly, to handle multiple training samples.
        bias_gradients = np.asarray([np.zeros(b.shape) for b in self._biases])
        weight_gradients = np.asarray([np.zeros(w.shape) for w in self._weights])
        for x, y in training_data:
            sample_bias_gradients, sample_weight_gradients = self._backpropagation(x, y)
            bias_gradients += np.asarray(sample_bias_gradients)
            weight_gradients += np.asarray(sample_weight_gradients)
        bias_gradients /= len(training_data)
        weight_gradients /= len(training_data)
        return bias_gradients, weight_gradients


    def _backpropagation(self, x, y):
        """
        Calculates the gradient of the cost function for a single training sample, using backpropagation.
        :param x: Input, shall be a column vector of the same length as the input layer.
        :param y: Expected output, shall be a column vector of the same length as the output layer.
        :return: Tuple containing bias gradients and weight gradients.
        """
        # Notations:
        #   c: The cost function for a single training sample,
        #   a: The neuron activation function,
        #   z: The neuron input function,
        #   dv_du: The derivative of v w.r.t u.
        bias_gradients = []
        weight_gradients = []

        # Feedforward step.
        # NOTE: This does the same thing as the feedforward method but also saves the output at each iteration.
        activations = [x]
        neuron_inputs = []
        for weights, biases in zip(self._weights, self._biases):
            z = np.dot(weights, x) + biases
            x = self._activation_function(z)
            neuron_inputs.append(z)
            activations.append(x)

        # Feedbackward step.
        # Calculate the derivative of the cost function w.r.t. the activation of the last layer dc_da.
        dc_da = self._cost_function_derivative(activations[-1], y)
        for layer in range(1, self._number_of_layers):
            # Calculate the derivative of the cost function w.r.t. the neuron inputs, dc_dz.
            da_dz = self._activation_function_derivative(neuron_inputs[-layer])
            dc_dz = np.multiply(dc_da, da_dz)  # Apply the chain rule.

            # Calculate the derivative of the cost function w.r.t. the biases, dc_db.
            dz_db = 1.0  # NOTE: Could be optimized sinze dz_db = 1.
            dc_db = np.multiply(dc_dz, dz_db)  # Apply the chain rule.
            bias_gradients.append(dc_db)

            # Calculate the derivative of the cost function w.r.t. the weights, dc_dw.
            dz_dw = activations[-layer - 1]
            # Use linear algebra to get all the weights and the correct shape.
            dc_dw = np.dot(dc_dz, dz_dw.transpose())  # Apply the chain rule
            weight_gradients.append(dc_dw)

            # Calculate the derivative of the cost function w.r.t. the activation of the previous layer, dc_da.
            dz_da = self._weights[-layer]
            # Use linear algebra to get all the activations and the correct shape.
            dc_da = np.dot(dz_da.transpose(), dc_dz)  # Apply the chain rule.

        return bias_gradients[::-1], weight_gradients[::-1]


    def _cost_function_derivative(self, network_output, expected_output):
        """
        The derivative of the cost function w.r.t. the network output for a single training sample, Cx'.
            C = 1/(n) * S(Cx),
            Cx = 1/2 * ||a - y||^2.
        :param network_output: The output of the network.
        :param expected_output: The expected output.
        :return: Cx'.
        """
        return network_output - expected_output
