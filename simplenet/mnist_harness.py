import numpy as np
from simple_machine import SimpleMachine


def mnist():
    from lib.mnist import read
    labels, images = read('training', path='../data')

    # normalize the pixel values; avoid overflow
    image_values = (images-128.0)/128.0
    label_values = np.unique(np.array(labels))
    result = (labels == label_values).astype(float)

    try:
        c = np.load('model_coefficients.dat.npy')
        s = SimpleMachine(image_values, result, [784, 10], c)
    except IOError:
        s = SimpleMachine(image_values, result, [784, 10])
    c = s.train(50000)
    np.save('model_coefficients.dat', c)


def mnist_predict():
    from lib.mnist import read
    labels, images = read('training', path='../data')

    # normalize the pixel values; avoid overflow
    image_values = (images-128.0)/128.0
    label_values = np.unique(np.array(labels))
    result = (labels == label_values)

    c = np.load('model_coefficients.dat.npy')
    s = SimpleMachine(image_values, result, [784, 10], c)
    predicted = s.score(image_values)
    m = np.amax(predicted, 1)
    actual_predicted = ((m * np.ones((1, 10))) == predicted)

    true_positives = np.sum(actual_predicted[result])
    print true_positives, labels.shape[0], 100 * true_positives/labels.shape[0]


if __name__ == "__main__":
    mnist_predict()
