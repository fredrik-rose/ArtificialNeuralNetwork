"""
The MNIST dataset contains images of handwritten digits and ground truth labels. The dataset is
split into two parts: 60 000 training samples and 10 000 test samples.
"""
import artificialneuralnetwork.input.idxparser as idx


IMAGE_RESOLUTION = 28 * 28


def load_mnist(images_path, labels_path):
    """
    Loads the MNIST dataset.
    :param images_path: Path to IDX file containing MNIST training or test images.
    :param labels_path: Path to IDX file containing the corresponding MNIST labels.
    :return: Dict containing the MNIST data; images and labels.
    """
    images = idx.parse(images_path) / 255
    labels = idx.parse(labels_path)
    return {'images': images, 'labels': labels}
