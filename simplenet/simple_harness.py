import numpy as np
from simple_machine import SimpleMachine


def simplest():
    t = np.matrix(((1.0, 0, 0, 0.99), (0, 0.8, 0, 0.95), (0, 0, 0.9, 0.9),
                   (1.0, 0, 0, 0.0), (0, 0.8, 0, 0.0), (0, 0, 0.9, 0.0)))
    l = [4, 3]
    r = np.matrix(((1, 0, 0), (0, 1, 0), (0, 0, 1),
                   (1, 0, 0), (0, 1, 0), (0, 0, 1)))
    s = SimpleMachine(t, r, l)
    s.train(1000)


if __name__ == "__main__":
    simplest()
