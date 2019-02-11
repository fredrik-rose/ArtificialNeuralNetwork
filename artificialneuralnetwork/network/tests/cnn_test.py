"""
CNN network unit tests.
"""
import numpy as np

import artificialneuralnetwork.network.cnn as cnn
import artificialneuralnetwork.network.tests.utilities as util


def test_fully_connected_gradient():
    """
    Checks the gradients of a fully connected network.
    """
    seed = 0
    layer_sizes = (7, 3, 3)
    number_of_training_samples = 5
    np.random.seed(seed)
    layers = []
    for input_size, output_size in zip(layer_sizes[:-1], layer_sizes[1:]):
        layers.append(cnn.FullyConnected(input_size, output_size))
        layers.append(cnn.Sigmoid())
    layers.pop()  # Special handling due to the cost function.
    cost = cnn.SingleClassCrossEntropy()
    network = cnn.NeuralNetworkModel(layers, cost)
    data = util.generate_training_data((layer_sizes[0], 1), (layer_sizes[-1], 1), number_of_training_samples, seed)
    _verify_gradients(network, data, seed)


def test_cnn_2d_single_gradient():
    """
    Checks the gradients of a CNN with no-depth input and no-depth output.
    """
    seed = 0
    number_of_training_samples = 7
    np.random.seed(seed)
    layers = [cnn.Convolutional(5, 1, 1),
              cnn.Sigmoid(),
              cnn.Convolutional(3, 1, 1),
              cnn.Sigmoid(),
              cnn.Convolutional(3, 1, 1),
              cnn.Flatten()]
    cost = cnn.MultiClassCrossEntropy()
    network = cnn.NeuralNetworkModel(layers, cost)
    data = util.generate_training_data((1, 30, 30), (22 * 22, 1), number_of_training_samples, seed)
    _verify_gradients(network, data, seed)


def test_cnn_3d_single_gradient():
    """
    Checks the gradients of a CNN with depth input and no-depth output.
    """
    seed = 0
    number_of_training_samples = 6
    np.random.seed(seed)
    layers = [cnn.Convolutional(5, 3, 1),
              cnn.Sigmoid(),
              cnn.Convolutional(3, 1, 1),
              cnn.Sigmoid(),
              cnn.Convolutional(3, 1, 1),
              cnn.Flatten()]
    cost = cnn.MultiClassCrossEntropy()
    network = cnn.NeuralNetworkModel(layers, cost)
    data = util.generate_training_data((3, 30, 30), (22 * 22, 1), number_of_training_samples, seed)
    _verify_gradients(network, data, seed, 1e-6)


def test_cnn_3d_multi_gradient():
    """
    Checks the gradients of a CNN with depth input and depth output.
    """
    seed = 0
    number_of_training_samples = 6
    np.random.seed(seed)
    layers = [cnn.Convolutional(5, 3, 3),
              cnn.Sigmoid(),
              cnn.Convolutional(3, 3, 5),
              cnn.Sigmoid(),
              cnn.Convolutional(1, 5, 7),
              cnn.Flatten()]
    cost = cnn.MultiClassCrossEntropy()
    network = cnn.NeuralNetworkModel(layers, cost)
    data = util.generate_training_data((3, 10, 10), (4 * 4 * 7, 1), number_of_training_samples, seed)
    _verify_gradients(network, data, seed, 1e-4)  # TODO: Maybe a bit large granularity


def test_relu_gradient():
    """
    Checks the gradients of the rectified linear unit (ReLU) layer.
    """
    seed = 0
    number_of_training_samples = 3
    np.random.seed(seed)
    layers = [cnn.Flatten(),
              cnn.FullyConnected(3 * 15 * 15, 10),
              cnn.ReLU()]
    cost = cnn.MultiClassCrossEntropy()
    network = cnn.NeuralNetworkModel(layers, cost)
    data = util.generate_training_data((3, 15, 15), (10, 1), number_of_training_samples, seed)
    # ReLU can cause problems due to the 'kink', i.e. not differentiable at 0.
    _verify_gradients(network, data, seed, 1e-5)


def test_cnn_max_pool_gradient():
    """
    Checks the gradients of the max pool layer.
    """
    seed = 0
    number_of_training_samples = 3
    np.random.seed(seed)
    layers = [cnn.Convolutional(3, 3, 5),
              cnn.Sigmoid(),
              cnn.MaxPool(2),
              cnn.Flatten()]
    cost = cnn.MultiClassCrossEntropy()
    network = cnn.NeuralNetworkModel(layers, cost)
    data = util.generate_training_data((3, 10, 10), (4 * 4 * 5, 1), number_of_training_samples, seed)
    # MaxPool can cause problems due to the 'kink', i.e. not differentiable at some points.
    _verify_gradients(network, data, seed, 1e-5)


def _verify_gradients(network, data, seed, granularity=0.5e-6):
    """
    Verifies gradients.
    :param network: Neural network.
    :param data: Data to run the network on.
    :param seed: Random seed.
    :param granularity: Granularity then comparing the analytical gradient with the numerical gradient.
    """
    def cost(network, data):
        """
        Creates a cost function from a neural network.
        :param network: Neural network.
        :param data: Network data.
        :return: Cost function.
        """
        def func():
            """
            Neural network cost function.
            :return: Cost.
            """
            np.random.seed(seed)
            return network._cost(data)

        return func

    biases = []
    weights = []
    for layer in network._layers:
        try:
            biases.append(layer._biases)
        except AttributeError:
            pass
        try:
            weights.append(layer._weights)
        except AttributeError:
            pass
    cost_func = cost(network, data)
    bias_gradients, weight_gradients = network._gradient(data)
    util.verify_function_gradients(cost_func, biases, bias_gradients, granularity)
    util.verify_function_gradients(cost_func, weights, weight_gradients, granularity)
