"""
Functionality that are used by many networks.
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


def sigmoid_derivative(z):
    """
    The derivative of the sigmoid function.
    :param z: The input.
    :return: Sigmoid'(z).
    """
    return np.exp(-z) / ((1 + np.exp(-z)) ** 2)


def asarray(array):
    """
    Converts the input to a numpy array.
    Does not try to merge dimensions as numpy's asarray function does.
    :param array: Input data.
    :return: Numpy array.
    """
    numpy_array = np.empty(len(array), dtype=object)
    for i, element in enumerate(array):
        numpy_array[i] = element
    return numpy_array
