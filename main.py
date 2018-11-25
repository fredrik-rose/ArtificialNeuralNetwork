"""
Program entry point.
"""
import argparse
import pickle

import artificialneuralnetwork.application.digitclassifier as dc
import artificialneuralnetwork.input.mnist as mnist

import config


def _save_object(obj, file_path):
    """
    Saves an object using pickle.
    :param obj: Object to save.
    :param file_path: Path to storage.
    """
    with open(file_path, 'wb') as file:  # NOTE: overwrites any existing file.
        pickle.dump(obj, file, pickle.HIGHEST_PROTOCOL)


def _load_object(file_path):
    """
    Loads an object using pickle.
    :param file_path: Path to load object from.
    :return: Loaded object.
    """
    with open(file_path, 'rb') as file:
        return pickle.load(file)


def _train(digit_classifier):
    """
    Trains a digit classifier.
    :param digit_classifier: The classifier.
    """
    mnist_dataset = mnist.load_mnist(config.MNIST_TRAIN_IMAGES_PATH, config.MNIST_TRAIN_LABELS_PATH)
    digit_classifier.train(mnist_dataset['images'], mnist_dataset['labels'])


def _evaluate(digit_classifier):
    """
    Evaluates a digit classifier.
    :param digit_classifier: The classifier.
    :return: TP (true positive) ratio [%]
    """
    mnist_dataset = mnist.load_mnist(config.MNIST_TEST_IMAGES_PATH, config.MNIST_TEST_LABELS_PATH)
    total = len(mnist_dataset['images'])
    correct = sum(digit_classifier.classify(image) == label
                  for image, label in zip(mnist_dataset['images'], mnist_dataset['labels']))
    return (correct / total) * 100


def main():
    """
    Main Function.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', help='train the classifier', dest='train', action='store_true')
    parser.add_argument('-e', help='evaluate the classifier', dest='evaluate', action='store_true')
    args = parser.parse_args()
    try:
        digit_classifier = _load_object(config.SAVED_DIGIT_CLASSIFIER_PATH)
    except FileNotFoundError:
        digit_classifier = dc.DigitClassifier(mnist.IMAGE_RESOLUTION)
    if args.train:
        _train(digit_classifier)
        _save_object(digit_classifier, config.SAVED_DIGIT_CLASSIFIER_PATH)
    if args.evaluate:
        true_positive_rate = _evaluate(digit_classifier)
        print("TP: {0:.2f}%".format(true_positive_rate))


if __name__ == "__main__":
    main()
