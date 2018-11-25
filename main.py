"""
Program entry point.
"""
import argparse
import pickle
import random as rnd

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


def _train(digit_classifier, train_dataset, test_dataset=None, epochs=30):
    """
    Trains a digit classifier.
    :param digit_classifier: The classifier.
    :param train_dataset: The train dataset.
    :param test_dataset: The test dataset.
    :param epochs: Number of training epochs.
    """
    if test_dataset is None:
        digit_classifier.train(train_dataset['images'], train_dataset['labels'], epochs)
    else:
        for epoch in range(epochs):
            digit_classifier.train(train_dataset['images'], train_dataset['labels'], 1)
            true_positive_rate = _evaluate(digit_classifier, test_dataset)
            print("TP after epoch {0:}: {1:.2f}%".format(epoch + 1, true_positive_rate))


def _evaluate(digit_classifier, test_dataset):
    """
    Evaluates a digit classifier.
    :param digit_classifier: The classifier.
    :param test_dataset: The dataset to evaluate on.
    :return: TP (true positive) ratio [%].
    """
    total = len(test_dataset['images'])
    correct = sum(digit_classifier.classify(image) == label
                  for image, label in zip(test_dataset['images'], test_dataset['labels']))
    return (correct / total) * 100


def _find_incorrectly_classified_images(digit_classifier, dataset):
    """
    Finds images with incorrect classification.
    :param digit_classifier: The classifier.
    :param dataset: Dataset containing images and labels.
    :return: The dataset containing only the incorrectly classifed images.
    """
    incorrect_classifications = {'images': [], 'labels': []}
    for image, label in zip(dataset['images'], dataset['labels']):
        classification = digit_classifier.classify(image)
        if classification != label:
            incorrect_classifications['images'].append(image)
            incorrect_classifications['labels'].append(classification)
    return incorrect_classifications


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


def main():
    """
    Main Function.
    """
    # TODO: This is a bit messy, clean up.
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', help='increase output verbosity', dest='verbose', action='store_true')
    parser.add_argument('-t', help='train the classifier', dest='train', action='store_true')
    parser.add_argument('-e', help='evaluate the classifier', dest='evaluate', action='store_true')
    parser.add_argument('-x', help='visualize images', dest='visualize', action='store_true')
    args = parser.parse_args()
    try:
        digit_classifier = _load_object(config.SAVED_DIGIT_CLASSIFIER_PATH)
    except FileNotFoundError:
        digit_classifier = dc.DigitClassifier(mnist.IMAGE_RESOLUTION)
    if args.train:
        mnist_train_dataset = mnist.load_mnist(config.MNIST_TRAIN_IMAGES_PATH, config.MNIST_TRAIN_LABELS_PATH)
        mnist_test_dataset = None
        digit_classifier = dc.DigitClassifier(mnist.IMAGE_RESOLUTION)
        if args.verbose:
            mnist_test_dataset = mnist.load_mnist(config.MNIST_TEST_IMAGES_PATH, config.MNIST_TEST_LABELS_PATH)
            print("Training on {} images, evaluating on {} images.".format(len(mnist_train_dataset['images']),
                                                                           len(mnist_test_dataset['images'])))
        if args.visualize:
            _visualize_dataset_subset(mnist_train_dataset, 'Subset of train images')
        _train(digit_classifier, mnist_train_dataset, mnist_test_dataset)
        _save_object(digit_classifier, config.SAVED_DIGIT_CLASSIFIER_PATH)
    if args.evaluate:
        mnist_test_dataset = mnist.load_mnist(config.MNIST_TEST_IMAGES_PATH, config.MNIST_TEST_LABELS_PATH)
        if args.verbose:
            print("Evaluating on {} images.".format(len(mnist_test_dataset['images'])))
        if args.visualize:
            incorrect_classifications = _find_incorrectly_classified_images(digit_classifier, mnist_test_dataset)
            _visualize_dataset_subset(mnist_test_dataset, 'Subset of evaluation images')
            _visualize_dataset_subset(incorrect_classifications, 'Subset of incorrectly classified images')
        true_positive_rate = _evaluate(digit_classifier, mnist_test_dataset)
        print("TP: {0:.2f}%".format(true_positive_rate))


if __name__ == "__main__":
    main()
