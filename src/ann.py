from __future__ import division, print_function

from abc import abstractmethod
from datetime import datetime


import random

from mnist_loader import load_data_wrapper

import numpy as np


class Network(object):
    def __init__(self, sizes, activations_function, cost, fast_mode=True):
        self.n_layers = len(sizes) - 1
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        self.activation_function = activations_function
        self.cost = cost
        self.fast_mode = fast_mode

    def feedforward(self, a):
        for w, b in zip(self.weights, self.biases):
            a = self.activation_function.apply(np.dot(w, a) + b)
        return a

    def sgd(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        if test_data:
            n_test = len(test_data)

        n = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size] for k in xrange(0, n, mini_batch_size)
            ]
            for mini_batch in mini_batches:
                if self.fast_mode:
                    self.matrix_update_mini_batch(mini_batch, eta)
                else:
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

        :type x: np.multiarray.ndarray
        :type y: np.multiarray.ndarray
        :return:
        """
        a = x
        activations = [a]
        zs = []

        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, a) + b
            zs.append(z)
            a = self.activation_function.apply(z)
            activations.append(a)

        deltas = [np.zeros([]) for _ in range(len(self.weights))]
        deltas[-1] = self.cost.cost_derivative(y, activations[-1], self.weights) * self.activation_function.derivative(
            zs[-1]
        )

        for i in reversed(range(len(deltas) - 1)):
            deltas[i] = np.dot(self.weights[i+1].T, deltas[i+1]) * self.activation_function.derivative(
                zs[i]
            )

        dc_dw = []
        dc_db = []

        for i in range(0, self.n_layers):
            dc_dw.append(np.outer(deltas[i], activations[i]))
            dc_db.append(deltas[i])

        return dc_db, dc_dw

    def matrix_update_mini_batch(self, mini_batch, eta):
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        xs, ys = map(
            np.hstack,
            zip(*mini_batch)
        )
        delta_nabla_b, delta_nabla_w = self.matrix_backprop(xs, ys)
        nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
        nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (eta/len(mini_batch)*nw) for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta/len(mini_batch)*nb) for b, nb in zip(self.biases, nabla_b)]

    def matrix_backprop(self, x, y):
        """

        :type x: np.multiarray.ndarray
        :type y: np.multiarray.ndarray
        :return:
        """
        a = x
        activations = [a]
        zs = []

        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, a) + b
            zs.append(z)
            a = self.activation_function.apply(z)
            activations.append(a)

        deltas = [np.zeros([]) for _ in range(len(self.weights))]
        deltas[-1] = self.cost.cost_derivative(y, activations[-1], self.weights) * self.activation_function.derivative(
            zs[-1]
        )

        for i in reversed(range(len(deltas) - 1)):
            deltas[i] = np.dot(self.weights[i+1].T, deltas[i+1]) * self.activation_function.derivative(
                zs[i]
            )

        dc_dw = []
        dc_db = []

        for i in range(0, self.n_layers):
            # dc_dw.append(np.outer(deltas[i], activations[i]))
            dc_dw.append(
                np.einsum(
                    'ij, kj -> ik',
                    deltas[i],
                    activations[i]
                )
            )
            dc_db.append(
                deltas[i].sum(axis=1, keepdims=True)
            )

        return dc_db, dc_dw

    def evaluate(self, test_data):
        xs, ys = zip(*test_data)
        y_hats = [np.argmax(self.feedforward(x)) for x in xs]
        return sum(int(y_hat == y) for y_hat, y in zip(y_hats, ys))

    @staticmethod
    def squared_error_cost(y, a):
        return 1/2*((y - a)**2).sum()

    @staticmethod
    def cost_derivative(y, a):
        return a - y


class Activation(object):
    def __init__(self):
        pass

    @staticmethod
    @abstractmethod
    def apply(z):
        pass

    @staticmethod
    @abstractmethod
    def derivative(z):
        pass


class Sigmoid(Activation):
    def __init__(self):
        super(self.__class__, self).__init__()

    @staticmethod
    def apply(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def derivative(z):
        s = Sigmoid.apply(z)
        return s * (1 - s)


class Cost(object):
    @abstractmethod
    def cost(y, a, w):
        pass

    @abstractmethod
    def cost_derivative(y, a, w):
        pass


class SquaredErrorCost(Cost):
    @staticmethod
    def cost(y, a, w=None):
        return 1/2*((y - a)**2).sum()

    @staticmethod
    def cost_derivative(y, a, w=None):
        return a - y


# class L1RegularizationCost(Cost):
#     def __init__(self, lambda_):
#         self.lambda_ = lambda_
#
#     def cost(self, y, a, w):
#         return self.lambda_ * np.sum([w_.sum() for w_ in w])
#
#     def cost_derivative(self, y, a, w):
#         return self.lambda_ * np.sign(w)


class SumOfCosts(Cost):
    def __init__(self, costs):
        self.costs = costs

    def cost(self, y, a, w):
        cost_ = reduce(
            np.add,
            [cost.cost(y, a, w) for cost in self.costs]
        )
        return cost_

    def cost_derivative(self, y, a, w):
        cost_derivative_ = reduce(
            np.add,
            [cost.cost_derivative(y, a, w) for cost in self.costs]
        )
        return cost_derivative_


# class ReLU(Activation):
#     def __init__(self):
#         super(self.__class__, self).__init__()
#
#     @staticmethod
#     def apply(z):
#         return np.maximum(0, z)
#
#     @staticmethod
#     def derivative(z):
#         return np.where(z > 0, 1, 0)


def main():
    np.random.seed(0)
    data = load_data_wrapper()
    network = Network(
        [784, 30, 10],
        activations_function=Sigmoid(),
        cost=SumOfCosts([SquaredErrorCost()]),
        fast_mode=True)
    t0 = datetime.utcnow()
    network.sgd(
        training_data=data[0],
        test_data=data[1],
        epochs=5,
        mini_batch_size=12,
        eta=3
    )
    print("Total time fast {}".format(datetime.utcnow() - t0))

if __name__ == '__main__':
    main()
