"""
Digit classifier.
"""
import numpy as np

import artificialneuralnetwork.network.mlp as mlp


def _convert_image(image):
    """
    Converts an image to the input format of a neural network.
    :param image: Image to be converted.
    :return: Converted image.
    """
    converted_image = image.flatten()
    converted_image.shape = (len(converted_image), 1)
    return converted_image


def _convert_label(label):
    """
    Converts a ground truth label to the output format of a neural network.
    :param label: Label to be converted.
    :return: Converted label.
    """
    converted_label = np.zeros((10, 1))
    converted_label[label] = 1.0
    return converted_label


def learning_rate_decay(learning_rate, epoch):
    """
    A learning rate decay strategy.
    :param learning_rate: Current learning rate.
    :param epoch: Current epoch.
    :return: New learning rate.
    """
    return learning_rate if epoch % 20 else learning_rate / 2


def create_digit_classifier_with_mlp(image_width, image_height, hidden_layers=(100, 30)):
    """
    Creates a digit classifier with a multilayer perceptron (mlp) network as backbone classifier.
    :param image_width: Width of the images to be classified.
    :param image_height: Height of the images to be classified.
    :param hidden_layers: Number of neurons in the hidden layers of the neural network.
    :return: Digit classifier.
    """
    training_params = {'regularization_factor': 3.0,
                       'momentum': 0.2,
                       'droput': 0,
                       'learning_rate_decay': learning_rate_decay}
    network = mlp.NeuralNetwork((image_width * image_height,) + hidden_layers + (10,))
    classifier = DigitClassifier(_convert_image, _convert_label, network, training_params)
    return classifier


class DigitClassifier():
    """
    A classifier that recognizes handwritten digits.
    Uses mean subtraction to zero center the data and data normalization, which could potentially
    speed up the learning.
    """
    def __init__(self, input_converter, output_converter, network, training_params=None):
        """
        Creates a digit classifier.
        :param input_converter: Function that converts the input to the format expected by the network.
        :param output_converter: Functions that converts the expected output to the format expected by the network.
        :param network: Artificial neural network backbone classifier.
        :param training_params: Parameters to be send to the training method of the network.
        """
        self._input_converter = input_converter
        self._output_converter = output_converter
        self._network = network
        self._training_params = training_params if training_params else {}
        self._mean_pixel = 0
        self._std_pixel = 0

    def classify(self, image):
        """
        Classifies an image containing a handwritten image.
        :param image: The image.
        :return: The digit in the image.
        """
        output = self._network.feedforward(self._input_converter(self._preprocess_image(image)))
        return np.argmax(output)

    def train(self, images, labels, epochs, batch_size=20, learning_rate=0.1):
        """
        Trains the backend neural network to recognize handwritten digits.
        :param images: Training images containing handwritten digits.
        :param labels: Training ground truth labels.
        :param epochs: See the train method in the corresponding backbone network class.
        :param batch_size: See the train method in the corresponding backbone network class.
        :param learning_rate: See the train method in the corresponding backbone network class.
        :return: See the train method in the corresponding backbone network class.
        """
        self._mean_pixel = np.mean(images)
        self._std_pixel = np.std(images)
        training_data = [(self._input_converter(self._preprocess_image(x)), self._output_converter(y))
                         for x, y in zip(images, labels)]
        return self._network.train(training_data, epochs, batch_size, learning_rate, **self._training_params)

    def _preprocess_image(self, image):
        """
        Zero centers and normalizes an image.
        :param image: Image to preprocess.
        :return: Preprocessed image.
        """
        return (image - self._mean_pixel) / (self._std_pixel + 1e-5)
