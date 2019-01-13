"""
MLP network unit tests.
"""
import numpy as np

import artificialneuralnetwork.network.mlp as mlp


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
    np.random.seed(seed)
    regularization /= number_of_training_samples
    data = [(np.random.uniform(-1, 1, (layer_sizes[0], 1)), np.random.uniform(0.001, 0.999, (layer_sizes[-1], 1)))
            for _ in range(number_of_training_samples)]
    network = mlp.NeuralNetwork(layer_sizes)
    np.random.seed(seed)
    bias_gradients, weight_gradients = network._gradient(data, regularization, dropout)
    _verify_gradient_set(seed, data, network, network._biases, bias_gradients, regularization, dropout, granularity)
    _verify_gradient_set(seed, data, network, network._weights, weight_gradients, regularization, dropout, granularity)


def _verify_gradient_set(seed, data, network, layers, gradients, regularization, dropout, granularity, delta=1e-5):
    """
    Verifies a set, e.g. biases or weights, of gradients.
    :param seed: Random seed used when calculating the analytical gradient.
    :param data: Input to the network.
    :param network: Neural network.
    :param layers: Reference to the parameters of the network, e.g. biases or weights.
    :param gradients: The parameters' corresponding analytical gradients.
    :param regularization: Amount of regularization used when calculating the analytical gradient.
    :param dropout: Dropout probability used when calculating the analytical gradient.
    :param granularity: Granularity then comparing the analytical gradient with the numerical gradient.
    :param delta: Delta to be used when calculating the numerical gradient.
    """
    assert len(gradients) == len(layers)
    for layer, gradient in zip(layers, gradients):
        assert gradient.shape == layer.shape
        original_layer = np.copy(layer)
        for index, analytical_gradient in np.ndenumerate(gradient):
            layer[index] = original_layer[index] - delta
            np.random.seed(seed)
            left = network._cost(data, regularization, dropout)
            layer[index] = original_layer[index] + delta
            np.random.seed(seed)
            right = network._cost(data, regularization, dropout)
            layer[index] = original_layer[index]
            numerical_gradient = (right - left) / (2 * delta)
            if not analytical_gradient == numerical_gradient == 0:
                relative_error = abs(analytical_gradient - numerical_gradient) / max(abs(analytical_gradient),
                                                                                     abs(numerical_gradient))
                assert relative_error <= granularity
