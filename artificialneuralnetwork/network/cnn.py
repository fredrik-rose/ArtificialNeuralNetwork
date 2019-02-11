"""
Convolutional neural network (CNN).
"""
import abc


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
