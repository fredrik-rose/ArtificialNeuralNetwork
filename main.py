"""
Program entry point.
"""
import argparse
import pickle
import random as rnd
import sys

import matplotlib.pyplot as plt
import numpy as np

import artificialneuralnetwork.application.digitclassifier as dc
import artificialneuralnetwork.input.augmentation as aug
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


def _augment_mnist_dataset(mnist_dataset):
    """
    Augments a MNIST dataset by rotating the images randomly.
    :param mnist_dataset: MNIST dataset to augment.
    :return: MNIST dataset extended by a factor of two using augmentation.
    """
    mnist_dataset['images'] = np.append(mnist_dataset['images'],
                                        aug.rotate_images(mnist_dataset['images'], 20),
                                        axis=0)
    mnist_dataset['labels'] = np.append(mnist_dataset['labels'],
                                        mnist_dataset['labels'],
                                        axis=0)


def _train(digit_classifier, mnist_train_dataset, mnist_test_dataset, epochs=40):
    """
    Trains a digit classifier.
    :param digit_classifier: The classifier.
    :param mnist_train_dataset: MNIST train dataset.
    :param mnist_test_dataset: MNIST test dataset.
    :param epochs: Number of training epochs.
    :return: List of costs, list of train accuracies, list of test accuracies.
    """
    def _evaluate_epoch(dataset):
        """
        Evaluates the accuracy of the current epoch.
        :param dataset: Dataset to evaluate.
        :return: Accuracy.
        """
        correct, total = _evaluate(digit_classifier, dataset)
        return (correct / total) * 100

    costs = []
    train_accuracies = []
    test_accuracies = []
    for epoch in range(epochs):
        # Remove the last element at each iteration since it will be equal to the first element of the next iteration.
        costs = costs[:-1] + digit_classifier.train(mnist_train_dataset['images'], mnist_train_dataset['labels'], 1)
        train_accuracies.append(_evaluate_epoch(mnist_train_dataset))
        test_accuracies.append(_evaluate_epoch(mnist_test_dataset))
        print("Accuracy after epoch {0:}/{1}: {2:.2f}%".format(epoch + 1, epochs, test_accuracies[-1]), flush=True)
    return costs, train_accuracies, test_accuracies


def _evaluate(digit_classifier, mnist_dataset):
    """
    Evaluates a digit classifier.
    :param digit_classifier: The classifier.
    :param mnist_dataset: MNIST dataset.
    :return: Tuple containing number of correct classifications and total number of classifications.
    """
    total = len(mnist_dataset['images'])
    correct = sum(digit_classifier.classify(image) == label
                  for image, label in zip(mnist_dataset['images'], mnist_dataset['labels']))
    return correct, total


def _split_correct_and_incorrect(digit_classifier, mnist_dataset):
    """
    Splits the samples in a dataset to correctly classified and incorrectly classified.
    :param digit_classifier: The classifier.
    :param mnist_dataset: MNIST dataset.
    :return: Tuple containing correct and incorrect classifications.
    """
    correct = {'images': [], 'labels': []}
    incorrect = {'images': [], 'labels': []}
    for image, label in zip(mnist_dataset['images'], mnist_dataset['labels']):
        classification = digit_classifier.classify(image)
        if classification == label:
            correct['images'].append(image)
            correct['labels'].append(classification)
        else:
            incorrect['images'].append(image)
            incorrect['labels'].append(classification)
    return correct, incorrect


def _visualize_costs(costs):
    """
    Visualizes the costs.
    :param costs: Costs to visualize.
    """
    plt.plot(costs, '-o')
    plt.xlabel("Epoch")
    plt.ylabel("Cost")
    plt.title("Costs")
    plt.show()


def _visualize_accuracies(train_accuracies, test_accuracies):
    """
    Visualizes the train and test accuracies.
    :param train_accuracies: Train accuracies.
    :param test_accuracies: Test accuracies.
    """
    plt.plot(train_accuracies, '-o', label="Train")
    plt.plot(test_accuracies, '-o', label="Test")
    plt.ylim([0, 100])
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracies")
    plt.show()


def _visualize_correct_incorrect(digit_classifier, mnist_dataset):
    """
    Visualizes some subsets of the training and test images.
    :param digit_classifier: The classifier.
    :param mnist_dataset: MNIST dataset.
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

    correct, incorrect = _split_correct_and_incorrect(digit_classifier, mnist_dataset)
    _visualize_dataset_subset(incorrect, 'Subset of incorrectly classified images')
    _visualize_dataset_subset(correct, 'Subset of correctly classified images')


def main():
    """
    Main Function.
    """
    parser = argparse.ArgumentParser(description='Handwritten-digit classifier.')
    parser.add_argument('-t', help='train the classifier', dest='train', action='store_true')
    parser.add_argument('-n', help='use only every N:th training sample', type=int, default=1)
    parser.add_argument('-e', help='evaluate the classifier', dest='evaluate', action='store_true')
    parser.add_argument('-x', help='visualize images', dest='visualize', action='store_true')
    args = parser.parse_args()
    mnist_train_dataset = mnist.load_mnist(config.MNIST_TRAIN_IMAGES_PATH, config.MNIST_TRAIN_LABELS_PATH)
    mnist_train_dataset['images'] = mnist_train_dataset['images'][::args.n]
    mnist_train_dataset['labels'] = mnist_train_dataset['labels'][::args.n]
    mnist_test_dataset = mnist.load_mnist(config.MNIST_TEST_IMAGES_PATH, config.MNIST_TEST_LABELS_PATH)
    if args.train:
        print("Extending data using data augmentation...", flush=True)
        _augment_mnist_dataset(mnist_train_dataset)
        print("Data augmentation completed.", flush=True)
        print("Training started...", flush=True)
        digit_classifier = dc.DigitClassifier(mnist.IMAGE_RESOLUTION)
        costs, train_accuracies, test_accuracies = _train(digit_classifier, mnist_train_dataset, mnist_test_dataset)
        _save_object(digit_classifier, config.SAVED_DIGIT_CLASSIFIER_PATH)
        print("Training completed on {} images.".format(len(mnist_train_dataset['images'])), flush=True)
        if args.visualize:
            _visualize_costs(costs)
            _visualize_accuracies(train_accuracies, test_accuracies)
    try:
        digit_classifier = _load_object(config.SAVED_DIGIT_CLASSIFIER_PATH)
    except FileNotFoundError:
        sys.exit("ERROR: Could not find a trained classifier. Run again with -t.")
    if args.evaluate:
        correct, total = _evaluate(digit_classifier, mnist_test_dataset)
        print("Accuracy: {0:.2f}% [{1}/{2}]".format((correct / total) * 100, correct, total), flush=True)
    if args.visualize:
        _visualize_correct_incorrect(digit_classifier, mnist_test_dataset)


if __name__ == "__main__":
    main()
