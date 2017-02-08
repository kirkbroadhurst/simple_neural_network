"""
Inspired by akesling on github
https://gist.github.com/akesling/5358964

Works with files from
http://yann.lecun.com/exdb/mnist/
"""


import os
import struct
import numpy as np

"""
Loosely inspired by http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
which is GPL licensed.
"""


def read(data_set="training", path="."):
    """
    Python function for importing the MNIST data set.  It returns an iterator
    of 2-tuples with the first element being the label and the second element
    being a numpy.uint8 2D array of pixel data for the given image.
    """

    if data_set is "training":
        name_img = os.path.join(path, 'train-images.idx3-ubyte')
        name_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
    elif data_set is "testing":
        name_img = os.path.join(path, 't10k-images.idx3-ubyte')
        name_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')
    else:
        raise ValueError("Data set must be 'testing' or 'training'")

    # Load everything in some numpy arrays
    with open(name_lbl, 'rb') as label_file:
        _, _ = struct.unpack(">II", label_file.read(8))
        labels = np.matrix(np.fromfile(label_file, dtype=np.int8)).T

    with open(name_img, 'rb') as image_file:
        magic, num, rows, cols = struct.unpack(">IIII", image_file.read(16))
        images = np.matrix(np.fromfile(image_file, dtype=np.uint8).reshape(labels.shape[0], rows*cols))

    return labels, images
