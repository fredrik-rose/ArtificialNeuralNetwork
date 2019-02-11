"""
Convolutional neural network (CNN).
"""
import abc

from scipy import ndimage
from scipy import signal
import numpy as np

import artificialneuralnetwork.network.common as cm


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


class Sigmoid(Layer):
    """
    A sigmoid layer.
    """

    def __init__(self):
        """
        Creates a sigmoid layer.
        """
        self._input = None

    def feedforward(self, x):
        """
        See the Layer class.
        """
        self._input = x
        return cm.sigmoid(x)

    @Layer.non_parameter
    def backpropagate(self, dc):
        """
        See the Layer class.
        """
        return np.multiply(cm.sigmoid_derivative(self._input), dc)


class ReLU(Layer):
    """
    A rectified linear unit (ReLU) layer.
    """

    def __init__(self):
        """
        Creates a ReLU layer.
        """
        self._input = None

    def feedforward(self, x):
        """
        See the Layer class.
        """
        self._input = x
        return cm.relu(x)

    @Layer.non_parameter
    def backpropagate(self, dc):
        """
        See the Layer class.
        """
        return np.multiply(cm.relu_derivative(self._input), dc)


class Convolutional(Layer):
    """
    A convolutional layer. Note that the input is not zero padded, i.e. the borders will be cut, the
    kernel size decides how much.
    """

    def __init__(self, kernel_size, input_depth, filters):
        """
        Creates a convolutional layer.
        :param kernel_size: The kernel (also known as filter) size of the layer.
        :param input_depth: The depth of the input to the layer.
        :param filters: Number of filters, i.e. the depth of the output.
        """
        self._weights = np.random.randn(filters, input_depth, kernel_size, kernel_size)
        self._biases = np.random.randn(filters, 1)
        self._input = None

    def feedforward(self, x):
        """
        See the Layer class. Could be implemented efficiently as a matrix multiplication using
        ol2img, see the cs231 course notes or
        https://fdsmlhn.github.io/2017/11/02/Understanding%20im2col%20implementation%20in%20Python(numpy%20fancy%20indexing)/
        """
        self._input = x
        output = [_remove_dimension(signal.convolve(x, filter, mode='valid')) + bias
                  for filter, bias in zip(self._weights, self._biases)]
        output = np.stack(output, axis=0)
        return output

    def backpropagate(self, dc, db, dw):
        """
        See the Layer class.
        """
        def calculate_local_db():
            """
            Calculates the derivative of the bias w.r.t. the cost function for this layer.
            :return: The derivative.
            """
            local_db = [np.sum(c) for c in dc]
            local_db = np.reshape(local_db, (len(local_db), 1))
            return local_db

        def calculate_local_dw():
            """
            Calculates the derivative of the weight w.r.t. the cost function for this layer.
            :return: The derivative.
            """
            local_dw = [signal.correlate(_add_dimension(c), self._input, mode='valid') for c in dc]
            local_dw = np.stack(local_dw, axis=0)
            return local_dw

        def calculate_local_dc():
            """
            Calculates the derivative of the cost function w.r.t. the input of this layer.
            :return: The derivative.
            """
            local_dc = [signal.correlate(_add_dimension(c), w, mode='full') for c, w in zip(dc, self._weights)]
            local_dc = np.sum(local_dc, axis=0)
            return local_dc

        db.append(calculate_local_db())
        dw.append(calculate_local_dw())
        return calculate_local_dc()

    def adjust(self, bias_delta, weight_delta):
        """
        See the Layer class.
        """
        self._biases += bias_delta.pop()
        self._weights += weight_delta.pop()


def _add_dimension(arr):
    """
    Add a dimension to an array.
    :param arr: Array.
    :return: Array with an extra first dimension.
    """
    return np.reshape(arr, (1, *arr.shape))


def _remove_dimension(arr):
    """
    Removes the first dimension of an array.
    :param arr: Array with first dimension equal to 1.
    :return: Array with removed first dimension.
    """
    return np.reshape(arr, arr.shape[1:])


class MaxPool(Layer):
    """
    A max pool layer, often used after a convolutional layer to reduce the size.
    """

    def __init__(self, kernel_size):
        """
        Creates a max pool layer.
        :param kernel_size: The kernel (also known as filter) size of the layer.
        """
        self._kernel_size = kernel_size
        self._max_indexes_mask = None

    def feedforward(self, x):
        """
        See the Layer class.
        """
        output = ndimage.filters.maximum_filter(x, size=(1, self._kernel_size, self._kernel_size),
                                                mode='constant', cval=0.0, origin=-(self._kernel_size-1))
        downsampled_output = self._downsample(output)
        self._max_indexes_mask = 1 * (x == self._upsample(downsampled_output))
        return downsampled_output

    @Layer.non_parameter
    def backpropagate(self, dc):
        """
        See the Layer class.
        """
        upsampled_dc = self._upsample(dc)
        return np.multiply(upsampled_dc, self._max_indexes_mask)

    def _downsample(self, arr):
        """
        Downsamples an array according to the filter size of the layer.
        :param arr: Array to downsample.
        :return: Downsampled array.
        """
        return arr[:, ::self._kernel_size, ::self._kernel_size]

    def _upsample(self, arr):
        """
        Upsamples an array according to the filter size of the layer.
        :param arr: Array to upsample.
        :return: Upsampled array.
        """
        return arr.repeat(self._kernel_size, axis=1).repeat(self._kernel_size, axis=2)


class Flatten(Layer):
    """
    A layer that flattens its input. Typically used between a convolutional layer and a fully
    connected layer.
    """

    def __init__(self):
        """
        Creates a flatten layer.
        """
        self._shape = None

    def feedforward(self, x):
        """
        See the Layer class.
        """
        self._shape = x.shape
        output = np.ndarray.flatten(x)
        output.shape = (len(output), 1)
        return output

    @Layer.non_parameter
    def backpropagate(self, dc):
        """
        See the Layer class.
        """
        return np.reshape(dc, self._shape)


class Cost(abc.ABC):
    """
    The interface of a generic neural network cost (also known as loss) function.
    """

    def feedforward(self, x):
        """
        The last layer of the neural network.
        :param x: Input, shape shall match the shape of the output of the network.
        :return: Output of the last layer of the network. Note that this is not the cost.
        """
        pass

    @abc.abstractmethod
    def cost(self, a, y):
        """
        The cost function.
        :param a: Neural network output, shape shall match the y parameter.
        :param y: Expected output, shape shall match the a parameter.
        :return: The cost (also known as loss).
        """
        pass

    @abc.abstractmethod
    def cost_derivative(self, a, y):
        """
        The derivative of the cost function, including the feedforward part.
        :param a: Neural network output, shape shall match the y parameter.
        :param y: Expected output, shape shall match the a parameter.
        :return: The derivative of the cost function.
        """
        pass


class SingleClassCrossEntropy(Cost):
    """
    A single-class cross-entropy (also known as the negative log likelihood of the Bernoulli
    distribution) cost function.
    """

    def feedforward(self, x):
        """
        See the Cost class.
        """
        return cm.sigmoid(x)

    def cost(self, a, y):
        """
        See the Cost class.
        """
        return -np.sum(np.nan_to_num(y * np.log(a) + (1 - y) * np.log(1 - a)))

    def cost_derivative(self, a, y):
        """
        See the Cost class.
        """
        return a - y
