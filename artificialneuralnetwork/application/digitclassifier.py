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
    """
    def __init__(self, image_resolution, hidden_layers=(30,)):
        """
        Initializes the backend neural network.
        :param image_resolution: Number of pixels in the images to be classified.
        :param hidden_layers: Number of neurons in the hidden layers of the neural network.
        """
        self.network = mlp.NeuralNetwork((image_resolution,) + hidden_layers + (10,))


    def classify(self, image):
        """
        Classifies an image containing a handwritten image.
        :param image: The image.
        :return: The digit in the image.
        """
        output = self.network.feedforward(_convert_image(image))
        return np.argmax(output)


    def train(self, images, labels, epochs=30, batch_size=10, learning_rate=3.0):
        """
        Trains the backend neural network to recognize handwritten digits.
        :param images: Training images containing handwritten digits.
        :param labels: Training ground truth labels.
        :param epochs: See the train method in the NeuralNetwork class.
        :param batch_size: See the train method in the NeuralNetwork class.
        :param learning_rate: See the train method in the NeuralNetwork class.
        """
        training_data = [(_convert_image(x), _convert_label(y))
                         for x, y in zip(images, labels)]
        self.network.train(training_data, epochs, batch_size, learning_rate)
