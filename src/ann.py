from __future__ import division, print_function

from abc import abstractmethod, ABCMeta
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
        stopping_criteria,
        learning_rate,
        update_algorithm,
        weight_initializer=initialize_input_dim_normalized_weights,
        regularization=None,
    ):
        self.n_layers = len(sizes) - 1
        self.sizes = sizes
        self.stopping_criteria = stopping_criteria
        self.learning_rate = learning_rate
        self.update_algorithm = update_algorithm
        self.biases = self.initialize_biases(self.sizes)
        self.weights = weight_initializer(self.sizes)
        self.activation_function = activations_function
        self.cost = cost
        self.regularization = regularization
        self.log = defaultdict(list)

    @staticmethod
    def initialize_biases(sizes):
        return [np.random.randn(y, 1) for y in sizes[1:]]

    def feedforward(self, a):
        for w, b in zip(self.weights, self.biases):
            a = self.activation_function.apply(np.dot(w, a) + b)
        return a

    def sgd(self, training_data, validation_data, mini_batch_size, test_data=None):
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        epoch = 1
        while not self.stopping_criteria(self.log):
            learning_rate = self.learning_rate.next(self.log)
            self.log['learning_rate'].append(learning_rate)
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size] for k in xrange(0, n, mini_batch_size)
            ]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, learning_rate, len(training_data))

            self.log['training_data_accuracy'].append(self.evaluate(training_data) / len(training_data))
            self.log['validation_data_accuracy'].append(self.evaluate(validation_data) / len(validation_data))
            self.log['validation_cost'].append(self.total_cost(validation_data, aggregate_f=np.mean))
            self.log['training_cost'].append(self.total_cost(training_data, aggregate_f=np.mean))
            self.log['test_cost'].append(self.total_cost(test_data, aggregate_f=np.mean))
            if test_data:
                n_correct = self.evaluate(test_data)
                self.log['test_data_accuracy'].append(n_correct / len(test_data))
                print(
                    "Epoch {0}: {1} / {2}".format(
                        epoch, self.evaluate(test_data), n_test
                    )
                )
            else:
                print("Epoch {0} complete".format(epoch))
            epoch += 1

    def update_mini_batch(self, mini_batch, learning_rate, training_set_size):
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        xs, ys = map(
            np.hstack,
            zip(*mini_batch)
        )
        delta_nabla_b, delta_nabla_w = self.backprop(xs, ys)
        nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
        nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = self.update_algorithm(
            self.weights,
            training_set_size,
            learning_rate,
            len(mini_batch),
            nabla_w
        )
        self.biases = [b - (learning_rate/len(mini_batch)*nb) for b, nb in zip(self.biases, nabla_b)]

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

    def total_cost(self, data, aggregate_f=sum):
        xs, ys = zip(*data)
        y_hats = [self.feedforward(x) for x in xs]
        costs = [self.cost.cost(y, y_hat) for y, y_hat in zip(ys, y_hats)]
        return aggregate_f(costs)

    @staticmethod
    def squared_error_cost(y, a):
        return 1/2*((y - a)**2).sum()

    @staticmethod
    def cost_derivative(y, a):
        return a - y


class UpdateAlgorithm(object):

    __metaclass__ = ABCMeta

    def __call__(self, weights, training_set_size, learning_rate, mini_batch_size, nabla_w):
        return [w + delta_w for w, delta_w in zip(
            weights,
            self.delta_w(
                weights,
                training_set_size,
                learning_rate,
                mini_batch_size,
                nabla_w
            )
        )]

    @abstractmethod
    def delta_w(self, weights, training_set_size, learning_rate, mini_batch_size, nabla_w):
        pass


class BasicUpdateAlgorithm(UpdateAlgorithm):
    def delta_w(self, weights, training_set_size, learning_rate, mini_batch_size, nabla_w):
        return [-(learning_rate/mini_batch_size*nw) for nw in nabla_w]


class L1UpdateAlgorithm(object):
    def __init__(self, lmbda):
        self.lmbda = lmbda


    def delta_w(self, weights, training_set_size, learning_rate, mini_batch_size, nabla_w):
        return [
            -np.sign(w)*learning_rate*self.lmbda/training_set_size - learning_rate/mini_batch_size*nw for w, nw in
            zip(weights, nabla_w)
        ]


class L2UpdateAlgorithm(UpdateAlgorithm):
    def __init__(self, lmbda):
        self.lmbda = lmbda

    def delta_w(self, weights, training_set_size, learning_rate, mini_batch_size, nabla_w):
        return [
            -w*learning_rate*self.lmbda/training_set_size - learning_rate/mini_batch_size*nw for w, nw in
            zip(weights, nabla_w)
        ]


class Momentum(UpdateAlgorithm):
    def __init__(self, momentum, base_algorithm):
        self.momentum = momentum
        self.base_algorithm = base_algorithm
        self.velocity = None

    def delta_w(self, weights, training_set_size, learning_rate, mini_batch_size, nabla_w):
        if self.velocity is None:
            self.velocity = np.zeros_like(weights)
        self.velocity = [self.momentum*v + nabla_w for v, nabla_w in zip(self.velocity, self.base_algorithm.delta_w(
            weights,
            training_set_size,
            learning_rate,
            mini_batch_size,
            nabla_w
        ))]
        return self.velocity


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
        c = -y*np.log(a) - (1-y)*np.log(1-a)
        c[np.isinf(c) | np.isnan(c)] = 0.0
        return np.sum(c)

    @staticmethod
    def cost_derivative(y, a):
        pass

    @staticmethod
    def delta(y, a, z):
        return a - y


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


class ReLU(Activation):
    def __init__(self):
        super(self.__class__, self).__init__()

    @staticmethod
    def apply(z):
        return np.maximum(0, z)

    @staticmethod
    def derivative(z):
        return np.where(z > 0, 1, 0)


class StoppingCriteria(object):
    pass


class NEpochs(StoppingCriteria):
    def __init__(self, n_epochs):
        self.n_epochs = n_epochs
        self.counter = 0

    def __call__(self, log):
        stop = self.counter >= self.n_epochs
        self.counter += 1
        return stop


class NoDecreaseInNEpochs(StoppingCriteria):
    def __init__(self, monitor_parameter, max_epochs):
        self.monitor_parameter = monitor_parameter
        self.max_epochs = max_epochs

    def __call__(self, log):
        n_epochs = len(log[self.monitor_parameter])
        if n_epochs == 0:
            stop = self.max_epochs <= 0
        else:
            stop = n_epochs - (1 + np.argmin(log[self.monitor_parameter])) >= self.max_epochs
        return stop


class LearningRateDecreaseLimit(StoppingCriteria):
    def __init__(self, initial_learning_rate, limit):
        self.initial_learning_rate = initial_learning_rate
        self.limit = limit

    def __call__(self, log):
        if len(log['learning_rate']) == 0:
            stop = False
        elif log['learning_rate'][-1] / self.initial_learning_rate < self.limit:
            stop = True
        else:
            stop = False
        return stop


class LearningRate(object):
    pass


class FixedLearningRate(LearningRate):
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def __iter__(self):
        return self

    def next(self, log):
        return self.learning_rate


class HalfLRIfNoDecreaseInNEpochs(LearningRate):
    def __init__(self, monitor_parameter, max_epochs, initial_learning_rate):
        self.monitor_parameter = monitor_parameter
        self.max_epochs = max_epochs
        self.learning_rate = initial_learning_rate

    def __iter__(self):
        return self

    def next(self, log):
        if self.no_decrease_in_n_epochs(log):
            self.learning_rate = self.learning_rate / 2
        return self.learning_rate

    def no_decrease_in_n_epochs(self, log):
        if len(log[self.monitor_parameter]) == 0:
            no_decrease_in_n_epochs_ = False
        else:
            n_epochs = len(log[self.monitor_parameter])
            no_decrease_in_n_epochs_ = n_epochs - (1 + np.argmin(log[self.monitor_parameter])) >= self.max_epochs
        return no_decrease_in_n_epochs_



def main():
    pass
    # lmbdas = [0.01, 0.1, 1, 10]
    # logs = []
    # for l in lmbdas:
    #     logs.append(run(l))
    # map(
    #     lambda log: print(log['validation_data_accuracy']),
    #     logs
    # )
    # print("The best lambda was {}".format(
    #     lmbdas[
    #         np.argmax(
    #             map(
    #                 lambda log: log['validation_data_accuracy'],
    #                 logs
    #             )
    #         )
    #     ]
    # ))


def run():
    np.random.seed(0)
    data = load_data_wrapper()
    initial_learning_rate = 0.2
    network = Network(
        [784, 100, 10],
        activations_function=Sigmoid(),
        cost=CrossEntropy(),
        stopping_criteria=NEpochs(50),#LearningRateDecreaseLimit(
        #     initial_learning_rate=initial_learning_rate,
        #     limit=1/2
        # ),
        learning_rate=FixedLearningRate(0.01),#HalfLRIfNoDecreaseInNEpochs(
        #     monitor_parameter='validation_cost',
        #     max_epochs=1,
        #     initial_learning_rate=initial_learning_rate
        # ),
        update_algorithm=Momentum(momentum=0.5, base_algorithm=L2UpdateAlgorithm(lmbda=5)),
        weight_initializer=initialize_input_dim_normalized_weights
    )
    t0 = datetime.utcnow()
    network.sgd(
        training_data=data[0],
        validation_data=data[1],
        test_data=data[2],
        mini_batch_size=12,
    )
    print("Total time fast {}".format(datetime.utcnow() - t0))
    plot_stats(network)
    return network.log


def plot_stats(network):
    f1, axs = plt.subplots(1, 4)
    axs[0].plot(network.log['test_data_accuracy'], label='Test Accuracy')
    axs[0].plot(network.log['training_data_accuracy'], label='Training Accuracy')
    axs[0].plot(network.log['validation_data_accuracy'], label='Validation Accuracy')

    axs[1].hist(np.hstack(map(lambda w: w.reshape(-1), network.weights)).reshape(-1), bins=100)

    axs[2].plot(network.log['validation_cost'], label='Validation Cost')
    axs[2].plot(network.log['training_cost'], label='Training Cost')
    axs[2].plot(network.log['test_cost'], label='Test Cost')

    axs[3].plot(network.log['learning_rate'], label='Learning Rate')

    axs[0].legend()
    axs[2].legend()
    axs[3].legend()
    plt.savefig("/home/sjosund/Programming/NeuralNets/img/ReLU.png")
    plt.show()


if __name__ == '__main__':
    run()
