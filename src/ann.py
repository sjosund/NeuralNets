from __future__ import division, print_function


import random

from mnist_loader import load_data_wrapper

import numpy as np


class Network(object):
    def __init__(self, sizes):
        self.n_layers = len(sizes) - 1
        self.sizes = sizes
        self.biases = [np.random.randn(y) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        for w, b in zip(self.weights, self.biases):
            a = Network.sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        if test_data: n_test = len(test_data)

        n = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size] for k in xrange(0, n, mini_batch_size)
            ]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print(
                    "Epoch {0}: {1} / {2}".format(
                        j, self.evaluate(test_data), n_test
                    )
                )
            else:
                print("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (eta/len(mini_batch)*nw) for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta/len(mini_batch)*nb) for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """

        :param x: numpy.array
        :param y: numpy.array
        :return:
        """
        a = x
        activations = []
        activations.append(a)
        zs = []

        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, a) + b
            zs.append(z)
            a = Network.sigmoid(z)
            activations.append(a)

        deltas = [np.zeros([]) for _ in range(len(self.weights))]
        deltas[-1] = Network.cost_derivative(y, activations[-1]) * Network.sigmoid_derivative(
            activations[-1]
        )

        for i in range(self.n_layers - 2, 0, -1):
            deltas[i] = np.dot(self.weights[i+1], deltas[i+1]) * Network.sigmoid_derivative(
                activations[i]
            )

        dc_dw = []
        dc_db = []

        for i in range(0, self.n_layers):
            dc_dw.append(np.outer(deltas[i], activations[i]))
            dc_db.append(deltas[i])

        return dc_db, dc_dw

    @staticmethod
    def squared_error_cost(y, a):
        return 1/2*((y - a)**2).sum()

    @staticmethod
    def cost_derivative(y, a):
        return a - y

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(z))

    @staticmethod
    def sigmoid_derivative(z):
        s = Network.sigmoid(z)
        return s * (1 - s)


def main():
    data = load_data_wrapper()
    network = Network([784, 10, 10])
    network.SGD(
        training_data=data[0],
        epochs=1,
        mini_batch_size=10,
        eta=0.1
    )


if __name__ == '__main__':
    main()

















