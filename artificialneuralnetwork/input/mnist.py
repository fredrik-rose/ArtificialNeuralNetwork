"""
The MNIST dataset contains images of handwritten digits and ground truth labels. The dataset is
split into two parts: 60 000 training samples and 10 000 test samples.
"""
import random as rnd

import numpy as np

import artificialneuralnetwork.input.idxparser as idx


IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28
_NUMBER_OF_VALIDATIONS = 10000


def load_mnist(train_images_path, train_labels_path, test_images_path, test_labels_path):
    """
    Loads the MNIST dataset.
    :param train_images_path: Path to IDX file containing MNIST training images.
    :param train_labels_path: Path to IDX file containing the corresponding training MNIST labels.
    :param test_images_path: Path to IDX file containing MNIST test images.
    :param test_labels_path: Path to IDX file containing the corresponding test MNIST labels.
    :return: Dictionary containing the MNIST dataset split into train, validation and test.
    """
    train_dataset = list(zip(idx.parse(train_images_path), idx.parse(train_labels_path)))
    rnd.shuffle(train_dataset)
    all_train_images, all_train_labels = (np.array(e) for e in zip(*train_dataset))
    train_images = all_train_images[_NUMBER_OF_VALIDATIONS:]
    train_labels = all_train_labels[_NUMBER_OF_VALIDATIONS:]
    validation_images = all_train_images[:_NUMBER_OF_VALIDATIONS]
    validation_labels = all_train_labels[:_NUMBER_OF_VALIDATIONS]
    test_images = idx.parse(test_images_path)
    test_labels = idx.parse(test_labels_path)
    return {dataset: {'images': images, 'labels': labels}
            for dataset, images, labels in (('train', train_images, train_labels),
                                            ('validation', validation_images, validation_labels),
                                            ('test', test_images, test_labels))}
