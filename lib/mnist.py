"""
From akesling on github
https://gist.github.com/akesling/5358964
"""


import os
import struct
import numpy as np

"""
Loosely inspired by http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
which is GPL licensed.
"""


def read(dataset="training", path="."):
    """
    Python function for importing the MNIST data set.  It returns an iterator
    of 2-tuples with the first element being the label and the second element
    being a numpy.uint8 2D array of pixel data for the given image.
    """

    if dataset is "training":
        name_img = os.path.join(path, 'train-images-idx3-ubyte')
        name_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset is "testing":
        name_img = os.path.join(path, 't10k-images-idx3-ubyte')
        name_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError("Data set must be 'testing' or 'training'")

    # Load everything in some numpy arrays
    with open(name_lbl, 'rb') as label_file:
        _, _ = struct.unpack(">II", label_file.read(8))
        lbl = np.fromfile(label_file, dtype=np.int8)

    with open(name_img, 'rb') as image_file:
        magic, num, rows, cols = struct.unpack(">IIII", image_file.read(16))
        img = np.fromfile(image_file, dtype=np.uint8).reshape(len(lbl), rows, cols)

    def get_img(idx):
        return lbl[idx], img[idx]

    # Create an iterator which returns each image in turn
    for i in xrange(len(lbl)):
        yield get_img(i)
