"""
Multilayer perceptron (MLP) network.
"""
import random as rnd

import numpy as np


def _sigmoid(z):
    """
    The sigmoid function.
    Could e.g. be used as an activation function in an artificial neural network.
    :param z: The input.
    :return: Sigmoid(z).
    """
    return 1 / (1 + np.exp(-z))


def _sigmoid_derivative(z):
    """
    The derivative of the sigmoid function.
    :param z: The input.
    :return: Sigmoid'(z).
    """
    return np.exp(-z) / ((1 + np.exp(-z)) ** 2)


class NeuralNetwork():
    """
    A multilayer perceptron (MLP), a class of feedforward artificial neural network.
    The cost function C used is defined as (the last term is a L2 regularization)
      C = 1/n * Sx(1/2 * ||a - y||^2) + h * 1/n * Sw(1/2 * w^2),
    where
      a is the actual output from the network,
      y is the expected output from the network,
      n is the number of training samples,
      Sx is the sum over all training samples,
      ||v|| denotes the length of vector v,
      h is the regularization factor,
      w is the weight,
      Sw is the sum over all weights.
    The regularization term reduces overfitting since it punishes large weights. Smaller weights
    means a less complex function, where small changes in the input will not make a great impact.

    Notations:
        c: The cost for a single training sample, c = 1/2 * ||a - y||^2.
        a: The neuron activation, a = activation_function(z).
        z: The neuron input, z = w*a + b.
        b: The bias.
        w: The weight.
        dv_du: The derivative of v w.r.t u.
    """
    def __init__(self, layer_sizes):
        """
        Initialize weights and biases to random values. The values are taken from the normal
        distribution with mean 0 and variance 1/n, where n is the number of inputs to the
        corresponding neuron. The reason for the chose of the variance is to avoid saturating the
        activation function for neurons with many inputs.
        :param layer_sizes: Tuple containing the number of neurons in each layer, including input,
                            hidden and output layers. Example with 2 hidden layer: (3,4,3,1).
        """
        self._number_of_layers = len(layer_sizes)
        self._biases = np.asarray([np.random.randn(size, 1)
                                   for size in layer_sizes[1:]])
        self._weights = np.asarray([np.random.randn(size, input_size) / np.sqrt(input_size)
                                    for input_size, size in zip(layer_sizes[:-1], layer_sizes[1:])])
        self._activation_function = _sigmoid
        self._activation_function_derivative = _sigmoid_derivative


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


    def train(self, training_data, epochs, batch_size, learning_rate, regularization_factor):
        """
        Trains the network using stochastic gradient descent.
        :param training_data: List of training pairs (input, expected output), which must match the
                              size of the input and output layers, respectively.
        :param epochs: Number of training epochs, i.e. number of passes over all the training samples.
        :param batch_size: Number of training samples in a batch, used to estimate the gradient for
                           a single gradient descent step.
        :param learning_rate: The gradient descent step size.
        :param regularization_factor: The amount of regularization.
        """
        for _ in range(epochs):
            rnd.shuffle(training_data)
            for i in range(0, len(training_data), batch_size):
                batch = training_data[i:i + batch_size]
                bias_gradients, weight_gradients = self._gradient(batch)
                weight_decay = 1 - (learning_rate * (regularization_factor / len(training_data)))
                self._biases -= learning_rate * bias_gradients
                self._weights = (weight_decay * self._weights) - (learning_rate * weight_gradients)


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
        def calculate_dc_db(dc_dz):
            """
            Calculates the derivative of the cost function w.r.t. the biases, dc_db.
                dz_db = 1,
                dc_db = dc_dz * dz_db = dc_dz * 1.
            :param dc_dz: The derivative of the cost function w.r.t. the neuron input.
            :return: dc_db.
            """
            dz_db = 1.0
            dc_db = np.multiply(dc_dz, dz_db)  # Apply the chain rule. NOTE: could be optimized since dz_db = 1.
            return dc_db

        def calculate_dc_dw(dc_dz, activation):
            """
            Calculates the derivative of the cost function w.r.t. the weights, dc_dw.
                dz_dw = a,
                dc_dw = dc_dz * dz_dw = dc_dz * a.
            :param dc_dz: The derivative of the cost function w.r.t. the neuron input.
            :param activation: The activation of the previous layer.
            :return: dc_dw.
            """
            dz_dw = activation
            # Use linear algebra to get all the weights and the correct shape.
            dc_dw = np.dot(dc_dz, dz_dw.transpose())  # Apply the chain rule.
            return dc_dw

        def calculate_dc_da(dc_dz, weight):
            """
            Calculates the derivative of the cost function w.r.t. the activation of the previous layer, dc_da.
                dz_da = w,
                dc_da = dc_dz * dz_da = dc_dz * w.
            :param dc_dz: The derivative of the cost function w.r.t. the neuron input.
            :param weight: The weight.
            :return: dc_da.
            """
            dz_da = weight
            # Use linear algebra to get all the activations and the correct shape.
            dc_da = np.dot(dz_da.transpose(), dc_dz)  # Apply the chain rule.
            return dc_da

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

        bias_gradients = []
        weight_gradients = []

        dc_dz = self._cost_function_derivative(activations[-1], neuron_inputs[-1], y)

        bias_gradients.append(calculate_dc_db(dc_dz))
        weight_gradients.append(calculate_dc_dw(dc_dz, activations[-2]))

        for layer in range(2, self._number_of_layers):
            dc_da = calculate_dc_da(dc_dz, self._weights[-layer + 1])

            # Calculate the derivative of the cost function w.r.t. the neuron inputs, dc_dz.
            #     da_dz = activation_function'(z),
            #     dc_dz = dc_da * da_dz = dc_da * activation_function'(z).
            da_dz = self._activation_function_derivative(neuron_inputs[-layer])
            dc_dz = np.multiply(dc_da, da_dz)  # Apply the chain rule.

            bias_gradients.append(calculate_dc_db(dc_dz))
            weight_gradients.append(calculate_dc_dw(dc_dz, activations[-layer - 1]))

        return bias_gradients[::-1], weight_gradients[::-1]


    def _cost_function_derivative(self, network_output, neuron_input, expected_output):
        """
        Calculates the derivative of the cost function w.r.t. the neuron inputs of the last layer,
        z, for a single training sample, c.
            dc_da = a - y,
            da_dz = activation_function'(z),
            dc_dz = dc_da * da_dz = (a - y) * activation_function'(z).
        :param network_output: The output of the network, a.
        :param neuron_input: The input to the neurons of the last layer, z.
        :param expected_output: The expected output, y.
        :return: dc_dz.
        """
        dc_da = network_output - expected_output
        da_dz = self._activation_function_derivative(neuron_input)
        dc_dz = np.multiply(dc_da, da_dz)  # Apply the chain rule.
        return dc_dz
