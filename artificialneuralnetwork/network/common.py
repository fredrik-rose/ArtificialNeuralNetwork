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


def relu(z):
    """
    The rectified linear unit (ReLU) function.
    Could e.g. be used as an activation function in an artificial neural network.
    :param z: The input.
    :return: ReLU(z).
    """
    return np.maximum(0, z)


def relu_derivative(z):
    """
    The derivative of the rectified linear unit (ReLU) function.
    :param z: The input.
    :return: ReLU'(z).
    """
    return 1 * (z > 0)


def softmax(z):
    """
    The softmax function.
    :param z: The input.
    :return: Softmax(z).
    """
    exponents = np.exp(z - np.max(z))  # The purpose of the subtraction is the make it numerically stable.
    return exponents / np.sum(exponents)


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


def aslist(array):
    """
    Converts the first dimension of a numpy array to a list.
    :param array: Input data.
    :return: List of numpy arrays.
    """
    return [e for e in array]
