"""
Convolutional neural network (CNN).
"""
import abc

import numpy as np


class Layer(abc.ABC):
    """
    The interface of a generic neural network layer.

    Inference:
        1. Run the feedforward method

    Training:
        1. Run the feedforward method
        2. Run the backpropagate method, the parameter gradients will be appended to the db and dw lists
        3. Run the adjust method, the parameter adjustments will be pop:ed from the bias_delta and weight_delta lists
    """

    @abc.abstractmethod
    def feedforward(self, x):
        """
        Runs the layer on a given input.
        :param x: Input, shape shall match the shape of the layer.
        :return: Layer output.
        """
        pass

    @abc.abstractmethod
    def backpropagate(self, dc, db, dw):
        """
        Performs backpropagation through the layer, i.e. calculates the gradients of the layer. The
        feedforward method must be called before. The handling of the db and dw parameters is very
        flexible but requires great care and synchronization with the adjust method. With great
        power comes great responsibility.
        :param dc: The derivative of the cost function w.r.t. the layer output.
        :param db: List of bias gradients w.r.t the cost function to append to. If and only if the
                   layer appends a gradient the bias delta must be pop:ed by the adjust method.
        :param dw: List of weight gradients w.r.t the cost function to append to. If and only if the
                   layer appends a gradient the weight delta must be pop:ed by the adjust method.
        :return: The derivative of the cost function w.r.t the layers inputs.
        """
        pass

    def adjust(self, bias_delta, weight_delta):
        """
        Adjusts the biases and weights of the layers. The handling of the bias_delta and
        weight_delta parameters is very flexible but requires great care and synchronization with
        the backpropagate method. With great power comes great responsibility.
        :param bias_delta: List of bias deltas to add to the biases. If and only if the layer appended
                           a bias gradient in the backpropagate method this list must be pop:ed.
        :param weight_delta: List of weight deltas to add to the weights. If and only if the layer appended
                             a weight gradient in the backpropagate method this list must be pop:ed.
        """
        pass

    def non_parameter(backpropagation_method):
        """
        Decorator to use with the backpropagate method, if the layer does not have learnable parameters.
        :param backpropagation_method: The backpropagation method.
        :return: Non-parametric backpropagation method.
        """

        def wrapper(self, dc, db, dw):
            """
            Non-parametric backpropagation method.
            :param dc: See the backpropagate method.
            :param db: Not used.
            :param dw: Not used.
            :return: See the backpropagate method.
            """
            del db, dw
            return backpropagation_method(self, dc)

        return wrapper


class FullyConnected(Layer):
    """
    A fully connected layer.
    """

    def __init__(self, input_size, output_size):
        """
        Creates a fully connected layer. The parameters are initialize for good performance on the
        ReLU activation function. Should work good also for Sigmoid (the optimal would have been to
        remove the 2 in the weight initialization, and probably use a normal distribution for the
        biases).
        :param input_size: Number of inputs to the layer.
        :param output_size: Number of outputs of the layer.
        """
        self._biases = np.zeros((output_size, 1))
        self._weights = np.random.randn(output_size, input_size) * np.sqrt(2 / input_size)
        self._input = None

    def feedforward(self, x):
        """
        See the Layer class.
        """
        self._input = x
        return np.dot(self._weights, x) + self._biases

    def backpropagate(self, dc, db, dw):
        """
        See the Layer class.
        """
        db.append(dc)
        dw.append(np.dot(dc, self._input.transpose()))
        return np.dot(self._weights.transpose(), dc)

    def adjust(self, bias_delta, weight_delta):
        """
        See the Layer class.
        """
        self._biases += bias_delta.pop()
        self._weights += weight_delta.pop()
