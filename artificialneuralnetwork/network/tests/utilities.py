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
