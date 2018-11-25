"""
Application configuration (bad practice since this is unsafe but ok for this toy application).
"""
import os


_DIRNAME = os.path.dirname(__file__)
MNIST_TRAIN_IMAGES_PATH = os.path.join(_DIRNAME, 'Data/MNIST/train-images-idx3-ubyte/train-images.idx3-ubyte')
MNIST_TRAIN_LABELS_PATH = os.path.join(_DIRNAME, 'Data/MNIST/train-labels-idx1-ubyte/train-labels.idx1-ubyte')
MNIST_TEST_IMAGES_PATH = os.path.join(_DIRNAME, 'Data/MNIST/t10k-images-idx3-ubyte/t10k-images.idx3-ubyte')
MNIST_TEST_LABELS_PATH = os.path.join(_DIRNAME, 'Data/MNIST/t10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte')
