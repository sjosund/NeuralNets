from __future__ import division, print_function

from abc import abstractmethod
from collections import defaultdict
from datetime import datetime
import random

import matplotlib.pyplot as plt
# import seaborn

import numpy as np

from mnist_loader import load_data_wrapper


def initialize_input_dim_normalized_weights(sizes):
    return [
        np.random.randn(y, x)/np.sqrt(x) for x, y in
        zip(sizes[:-1], sizes[1:])
    ]


def initialize_large_weights(sizes):
    return [
        np.random.randn(y, x)/np.sqrt(x) for x, y in
        zip(sizes[:-1], sizes[1:])
    ]


class Network(object):
    def __init__(
        self,
        sizes,
        activations_function,
        cost,
        weight_initializer=initialize_input_dim_normalized_weights,
        fast_mode=True,
        regularization=None
    ):
        self.n_layers = len(sizes) - 1
        self.sizes = sizes
        self.biases = self.initialize_biases(self.sizes)
        self.weights = weight_initializer(self.sizes)
        self.activation_function = activations_function
        self.cost = cost
        self.fast_mode = fast_mode
        self.regularization = regularization
        self.log = defaultdict(list)

    @staticmethod
    def initialize_biases(sizes):
        return [np.random.randn(y, 1) for y in sizes[1:]]

    def feedforward(self, a):
        for w, b in zip(self.weights, self.biases):
            a = self.activation_function.apply(np.dot(w, a) + b)
        return a

    def sgd(self, training_data, epochs, mini_batch_size, eta, test_data=None, lmbda=5):
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
                    self.matrix_update_mini_batch(mini_batch, eta, lmbda, len(training_data))
                else:
                    raise Exception("Deprecated")
                #     self.update_mini_batch(mini_batch, eta, lmbda, len(training_data))

            self.log['training_data_accuracy'].append(self.evaluate(training_data) / len(training_data))
            if test_data:
                n_correct = self.evaluate(test_data)
                self.log['test_data_accuracy'].append(n_correct / len(test_data))
                print(
                    "Epoch {0}: {1} / {2}".format(
                        j, self.evaluate(test_data), n_test
                    )
                )
            else:
                print("Epoch {0} complete".format(j))

    # def update_mini_batch(self, mini_batch, eta, lmbda, n):
    #     nabla_w = [np.zeros(w.shape) for w in self.weights]
    #     nabla_b = [np.zeros(b.shape) for b in self.biases]
    #     for x, y in mini_batch:
    #         delta_nabla_b, delta_nabla_w = self.backprop(x, y)
    #         nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
    #         nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
    #     self.weights = [(1 - eta*(lmbda/n))*w - (eta/len(mini_batch)*nw) for w, nw in zip(self.weights, nabla_w)]
    #     self.biases = [b - (eta/len(mini_batch)*nb) for b, nb in zip(self.biases, nabla_b)]

    # def backprop(self, x, y):
    #     """
    #
    #     :type x: np.multiarray.ndarray
    #     :type y: np.multiarray.ndarray
    #     :return:
    #     """
    #     a = x
    #     activations = [a]
    #     zs = []
    #
    #     for w, b in zip(self.weights, self.biases):
    #         z = np.dot(w, a) + b
    #         zs.append(z)
    #         a = self.activation_function.apply(z)
    #         activations.append(a)
    #
    #     deltas = [np.zeros([]) for _ in range(len(self.weights))]
    #     deltas[-1] = self.cost.cost_derivative(y, activations[-1]) * self.activation_function.derivative(
    #         zs[-1]
    #     )
    #
    #     for i in reversed(range(len(deltas) - 1)):
    #         deltas[i] = np.dot(self.weights[i+1].T, deltas[i+1]) * self.activation_function.derivative(
    #             zs[i]
    #         )
    #
    #     dc_dw = []
    #     dc_db = []
    #
    #     for i in range(0, self.n_layers):
    #         dc_dw.append(np.outer(deltas[i], activations[i]))
    #         dc_db.append(deltas[i])
    #
    #     return dc_db, dc_dw

    def matrix_update_mini_batch(self, mini_batch, eta, lmbda, n):
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        xs, ys = map(
            np.hstack,
            zip(*mini_batch)
        )
        delta_nabla_b, delta_nabla_w = self.matrix_backprop(xs, ys)
        nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
        nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        if self.regularization is None:
            self.weights = [w - (eta/len(mini_batch)*nw) for w, nw in zip(self.weights, nabla_w)]
        elif self.regularization == 'L1':
            self.weights = [w - np.sign(w)*eta*lmbda/n - eta/len(mini_batch)*nw for w, nw in zip(self.weights, nabla_w)]
        elif self.regularization == 'L2':
            self.weights = [(1 - eta*lmbda/n)*w - (eta/len(mini_batch)*nw) for w, nw in zip(self.weights, nabla_w)]
        else:
            raise NotImplementedError("Regularization method not implemented")
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
        deltas[-1] = self.cost.delta(y, activations[-1], zs[-1])#cost_derivative(y, activations[-1]) * self.activation_function.derivative(
            # zs[-1]
        # )

        for i in reversed(range(len(deltas) - 1)):
            deltas[i] = np.dot(
                self.weights[i+1].T, deltas[i+1]
            ) * self.activation_function.derivative(
                zs[i]
            )

        dc_dw = []
        dc_db = []

        for i in range(0, self.n_layers):
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
        return sum(int(y_hat == np.argmax(y)) for y_hat, y in zip(y_hats, ys))

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
    def cost(y, a):
        pass

    @abstractmethod
    def cost_derivative(y, a):
        pass


class SquaredErrorCost(Cost):
    @staticmethod
    def cost(y, a):
        return 1/2*((y - a)**2).sum()

    @staticmethod
    def cost_derivative(y, a):
        return a - y

    @staticmethod
    def delta(y, a, z):
        raise NotImplementedError()


class CrossEntropy(Cost):
    @staticmethod
    def cost(y, a):
        return np.sum(np.nan_to_num(y*np.log(a) + (1-y)*np.log(1-a)))

    @staticmethod
    def cost_derivative(y, a):
        pass

    @staticmethod
    def delta(y, a, z):
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


# class SumOfCosts(Cost):
#     def __init__(self, costs):
#         self.costs = costs
#
#     def cost(self, y, a):
#         cost_ = reduce(
#             np.add,
#             [cost.cost(y, a) for cost in self.costs]
#         )
#         return cost_
#
#     def cost_derivative(self, y, a):
#         cost_derivative_ = reduce(
#             np.add,
#             [cost.cost_derivative(y, a) for cost in self.costs]
#         )
#         return cost_derivative_


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
        [784, 100, 10],
        activations_function=Sigmoid(),
        cost=CrossEntropy(),
        weight_initializer=initialize_input_dim_normalized_weights,
        fast_mode=True,
        regularization='L2'
    )
    t0 = datetime.utcnow()
    network.sgd(
        training_data=data[0],
        test_data=data[1],
        epochs=15,
        mini_batch_size=12,
        eta=0.5
    )
    print("Total time fast {}".format(datetime.utcnow() - t0))
    plot_stats(network)


def plot_stats(network):
    f1, axs = plt.subplots(1, 2)
    axs[0].plot(network.log['test_data_accuracy'], label='Test Accuracy')
    axs[0].plot(network.log['training_data_accuracy'], label='Training Accuracy')
    axs[1].hist(np.hstack(map(lambda w: w.reshape(-1), network.weights)).reshape(-1), bins=10000)
    plt.legend()
    plt.savefig("/home/sjosund/Programming/NeuralNets/img/L2.png")
    plt.show()


if __name__ == '__main__':
    main()
