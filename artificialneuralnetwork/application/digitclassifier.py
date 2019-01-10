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


class DigitClassifier():
    """
    A classifier that recognizes handwritten digits.
    Uses mean subtraction to zero center the data and data normalization, which could potentially
    speed up the learning.
    """
    def __init__(self, image_resolution, hidden_layers=(100, 30)):
        """
        Initializes the backend neural network.
        :param image_resolution: Number of pixels in the images to be classified.
        :param hidden_layers: Number of neurons in the hidden layers of the neural network.
        """
        self._mean_pixel = 0
        self._std_pixel = 0
        self._network = mlp.NeuralNetwork((image_resolution,) + hidden_layers + (10,))

    def classify(self, image):
        """
        Classifies an image containing a handwritten image.
        :param image: The image.
        :return: The digit in the image.
        """
        output = self._network.feedforward(_convert_image(self._preprocess_image(image)))
        return np.argmax(output)

    def train(self, images, labels, epochs, batch_size=20, learning_rate=0.1, regularization_factor=3.0, momentum=0.2,
              droput=0, learning_rate_decay=lambda r, e: r if e % 20 else r / 2):
        """
        Trains the backend neural network to recognize handwritten digits.
        :param images: Training images containing handwritten digits.
        :param labels: Training ground truth labels.
        :param epochs: See the train method in the NeuralNetwork class.
        :param batch_size: See the train method in the NeuralNetwork class.
        :param learning_rate: See the train method in the NeuralNetwork class.
        :param regularization_factor: See the train method in the NeuralNetwork class.
        :param momentum: See the train method in the NeuralNetwork class.
        :param droput: See the train method in the NeuralNetwork class.
        """
        self._mean_pixel = np.mean(images)
        self._std_pixel = np.std(images)
        training_data = [(_convert_image(self._preprocess_image(x)), _convert_label(y))
                         for x, y in zip(images, labels)]
        self._network.train(training_data, epochs, batch_size, learning_rate, regularization_factor, momentum, droput,
                            learning_rate_decay)

    def _preprocess_image(self, image):
        """
        Zero centers and normalizes an image.
        :param image: Image to preprocess.
        :return: Preprocessed image.
        """
        return (image - self._mean_pixel) / (self._std_pixel + 1e-5)
