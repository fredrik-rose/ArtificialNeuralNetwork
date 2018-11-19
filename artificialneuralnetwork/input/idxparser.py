"""
IDX file parser.
"""
import struct

import numpy as np


def parse(path):
    """
    Parses an IDX file to a numpy ndarray.

    NOTE: Performs no error checks whatsoever.

    :param path: Path to the idx file.
    :return: Numpy ndarray.
    """
    data_types = {
        0x08: np.uint8,  # Unsigned byte
        0x09: np.int8,  # Signed byte
        0x0B: np.int16,  # Short
        0x0C: np.int32,  # Int
        0x0D: np.float32,  # Float
        0x0E: np.float64,  # Double
    }
    with open(path, 'rb') as file:
        file.read(2)  # The first two bytes are always zero.
        data_type, dimensions = struct.unpack('BB', file.read(2))
        # The data is in big-endian order.
        dimension_sizes = struct.unpack('>{}'.format('I' * dimensions), file.read(dimensions * 4))
        data = np.fromfile(file, dtype=np.dtype(data_types[data_type]).newbyteorder('>'))
        return data.reshape(dimension_sizes)
