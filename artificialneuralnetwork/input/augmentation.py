"""
Data augmentation.
Use these functions to extend the dataset which could reduce overfitting.
"""
import random

from scipy import ndimage


def rotate_images(images, max_rotation):
    """
    Performs data augmentation by randomly rotating images.
    :param images: Images to augment.
    :param max_rotation: Max rotation angle in either direction.
    :return: Augmented images.
    """
    return [ndimage.rotate(image, angle=random.uniform(-max_rotation, max_rotation), reshape=False)
            for image in images]
