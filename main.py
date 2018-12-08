"""
Program entry point.
"""
import argparse
import pickle
import random as rnd
import sys

import matplotlib.pyplot as plt

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


def _train(digit_classifier, evaluate=False, epochs=40):
    """
    Trains a digit classifier.
    :param digit_classifier: The classifier.
    :param evaluate: Evaluates after each training epoch if true.
    :param epochs: Number of training epochs.
    :return: Number of training images.
    """
    mnist_train_dataset = mnist.load_mnist(config.MNIST_TRAIN_IMAGES_PATH, config.MNIST_TRAIN_LABELS_PATH)
    if evaluate:
        for epoch in range(epochs):
            digit_classifier.train(mnist_train_dataset['images'], mnist_train_dataset['labels'], 1)
            correct, total = _evaluate(digit_classifier)
            print("TP after epoch {0:}/{1}: {2:.2f}%".format(epoch + 1, epochs, (correct / total) * 100), flush=True)
    else:
        digit_classifier.train(mnist_train_dataset['images'], mnist_train_dataset['labels'], epochs)
    return len(mnist_train_dataset['images'])


def _evaluate(digit_classifier):
    """
    Evaluates a digit classifier.
    :param digit_classifier: The classifier.
    :return: Tuple containing number of correct classifications and total number of classifications.
    """
    mnist_test_dataset = mnist.load_mnist(config.MNIST_TEST_IMAGES_PATH, config.MNIST_TEST_LABELS_PATH)
    total = len(mnist_test_dataset['images'])
    correct = sum(digit_classifier.classify(image) == label
                  for image, label in zip(mnist_test_dataset['images'], mnist_test_dataset['labels']))
    return correct, total


def _split_correct_and_incorrect(digit_classifier, dataset):
    """
    Splits the samples in a dataset to correctly classified and incorrectly classified.
    :param digit_classifier: The classifier.
    :param dataset: Dataset containing images and labels.
    :return: Tuple containing correct and incorrect classifications.
    """
    correct = {'images': [], 'labels': []}
    incorrect = {'images': [], 'labels': []}
    for image, label in zip(dataset['images'], dataset['labels']):
        classification = digit_classifier.classify(image)
        if classification == label:
            correct['images'].append(image)
            correct['labels'].append(classification)
        else:
            incorrect['images'].append(image)
            incorrect['labels'].append(classification)
    return correct, incorrect


def _visualize(digit_classifier):
    """
    Visualizes some subsets of the training and test images.
    :param digit_classifier: The classifier.
    """
    def _visualize_dataset_subset(dataset, title='', rows=4, cols=5):
        """
        Visualizes a subset of a dataset of images and labels.
        :param dataset: The dataset.
        :param title: Title of the plot.
        :param rows: Number of rows with images in the plot.
        :param cols: Number of columns of images in the plot.
        """
        image_label_pairs = [(image, label) for image, label in zip(dataset['images'], dataset['labels'])]
        rnd.shuffle(image_label_pairs)
        for index, (image, label) in enumerate(image_label_pairs[:rows * cols]):
            plt.subplot(rows, cols, index + 1)
            plt.imshow(image, cmap='gray')
            plt.title(label)
        plt.suptitle(title)
        plt.show()

    mnist_test_dataset = mnist.load_mnist(config.MNIST_TEST_IMAGES_PATH, config.MNIST_TEST_LABELS_PATH)
    correct, incorrect = _split_correct_and_incorrect(digit_classifier, mnist_test_dataset)
    _visualize_dataset_subset(incorrect, 'Subset of incorrectly classified images')
    _visualize_dataset_subset(correct, 'Subset of correctly classified images')


def main():
    """
    Main Function.
    """
    parser = argparse.ArgumentParser(description='Handwritten-digit classifier.')
    parser.add_argument('-t', help='train the classifier', dest='train', action='store_true')
    parser.add_argument('-e', help='evaluate the classifier', dest='evaluate', action='store_true')
    parser.add_argument('-x', help='visualize images', dest='visualize', action='store_true')
    args = parser.parse_args()
    if args.train:
        print("Training started...", flush=True)
        digit_classifier = dc.DigitClassifier(mnist.IMAGE_RESOLUTION)
        total = _train(digit_classifier, args.evaluate)
        _save_object(digit_classifier, config.SAVED_DIGIT_CLASSIFIER_PATH)
        print("Training completed on {} images.".format(total), flush=True)
    try:
        digit_classifier = _load_object(config.SAVED_DIGIT_CLASSIFIER_PATH)
    except FileNotFoundError:
        sys.exit("ERROR: Could not find a trained classifer. Run again with -t.")
    if args.evaluate:
        correct, total = _evaluate(digit_classifier)
        print("TP: {0:.2f}% [{1}/{2}]".format((correct / total) * 100, correct, total), flush=True)
    if args.visualize:
        _visualize(digit_classifier)


if __name__ == "__main__":
    main()
