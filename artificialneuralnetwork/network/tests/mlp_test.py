"""
MLP network unit tests.
"""
import numpy as np

import artificialneuralnetwork.network.mlp as mlp
import artificialneuralnetwork.network.tests.utilities as util


def test_gradient():
    """
    Checks the gradients.
    """
    seed = 0
    layer_sizes = (7, 3)
    number_of_training_samples = 1
    _verify_gradients(seed, layer_sizes, number_of_training_samples, 0, 0)


def test_gradient_with_regularization():
    """
    Checks the gradients when using regularization.
    """
    seed = 100
    layer_sizes = (5, 3, 4)
    number_of_training_samples = 3
    regularization = 100
    _verify_gradients(seed, layer_sizes, number_of_training_samples, regularization, 0)


def test_gradient_with_dropout():
    """
    Checks the gradients when using dropout.
    """
    seed = 230
    layer_sizes = (7, 2, 2, 2)
    number_of_training_samples = 5
    dropout = 0.5
    _verify_gradients(seed, layer_sizes, number_of_training_samples, 0, dropout)


def test_gradient_with_regularization_and_dropout():
    """
    Checks the gradients when using regularization and dropout.
    """
    seed = 1024
    layer_sizes = (5, 5, 5, 5)
    number_of_training_samples = 42
    regularization = 10
    dropout = 0.3
    _verify_gradients(seed, layer_sizes, number_of_training_samples, regularization, dropout)


def _verify_gradients(seed, layer_sizes, number_of_training_samples, regularization, dropout, granularity=0.5e-6):
    """
    Verifies gradients.
    :param seed: Random seed.
    :param layer_sizes: Tuple containing the layer sizes of the network.
    :param number_of_training_samples: Number of training samples.
    :param regularization: Amount of regularization.
    :param dropout: Dropout probability.
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
            return network._cost(data, regularization, dropout)

        return func

    regularization /= number_of_training_samples
    data = util.generate_training_data((layer_sizes[0], 1), (layer_sizes[-1], 1), number_of_training_samples, seed)
    np.random.seed(seed)
    network = mlp.NeuralNetwork(layer_sizes)
    cost_func = cost(network, data)
    np.random.seed(seed)
    bias_gradients, weight_gradients = network._gradient(data, regularization, dropout)
    util.verify_function_gradients(cost_func, network._biases, bias_gradients, granularity)
    util.verify_function_gradients(cost_func, network._weights, weight_gradients, granularity)
