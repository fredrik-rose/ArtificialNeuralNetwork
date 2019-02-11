"""
Test utilities.
"""
import numpy as np


def generate_training_data(input_size, output_size, number_of_training_samples, seed=0):
    """
    Generates (random) training data for a neural network.
    :param input_size: Input size of the network.
    :param output_size: Output size of the network.
    :param number_of_training_samples: Number of training samples to generate.
    :param seed: Random generator seed.
    :return: Training data [(input, output), ...].
    """
    np.random.seed(seed)
    return [(np.random.uniform(-1, 1, input_size), np.random.uniform(0.001, 0.999, output_size))
            for _ in range(number_of_training_samples)]


def verify_function_gradients(func, parameters, gradients, granularity, delta=1e-5):
    """
    Verifies the analytical gradients of a function by comparing with the numerical gradients. Note
    that the function must be differentiable in all points for a reliable behavior.
    :param func: The function, must be depended on the parameters.
    :param parameters: The parameters of the function, a change must affect the function.
    :param gradients: The analytical gradients of the function.
    :param granularity: Max relative error.
    :param delta: Step size when calculating the numerical gradients.
    """
    assert len(gradients) == len(parameters)
    for layer, gradient in zip(parameters, gradients):
        assert gradient.shape == layer.shape
        original_layer = np.copy(layer)
        for index, analytical_gradient in np.ndenumerate(gradient):
            layer[index] = original_layer[index] - delta
            left = func()
            layer[index] = original_layer[index] + delta
            right = func()
            layer[index] = original_layer[index]
            numerical_gradient = (right - left) / (2 * delta)
            error = _relative_error(analytical_gradient, numerical_gradient)
            assert error <= granularity


def _relative_error(actual, expected):
    """
    Calculates the relative error.
    :param actual: Actual value.
    :param expected: Expected value.
    :return: Relative error.
    """
    if actual == 0 or expected == 0:
        return abs(actual - expected)
    return abs(actual - expected) / max(abs(actual), abs(expected))
