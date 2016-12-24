import csv
import numpy as np


def load_csv(file_name):
    """
    Load a csv file into a numpy matrix of floats
    :param file_name: file name
    :return:
    """
    with open(file_name, 'r') as f:
        data_iter = csv.reader(f, delimiter=',')
        data = [data for data in data_iter]
    matrix = np.matrix(data, dtype=float)
    return matrix
