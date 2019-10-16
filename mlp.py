""" Multi Layer Perceptron
Author: Guilherme Casimiro """

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import ipdb
from utilities import mlp_train_val_test
from utilities import tanh_derivative
from sklearn.datasets.samples_generator import make_blobs


class MLP(object):
    """docstring for Perceptron"""
    def __init__(self, input_num, h_num, output_num, alpha, epoch_max):
        self.alpha = alpha
        self.epoch_max = epoch_max
        self.i_weights = np.random.rand(input_num + 1, h_num + 1)
        self.o_weights = np.random.rand(h_num + 1, output_num)
        self.mse = np.inf

    def feedforward(self, x):
        self.h_layer = np.tanh(np.dot(x, self.i_weights))
        output = np.dot(self.h_layer, self.o_weights)
        return output

    def backpropagation(self, X, error):
        d_o_weights = np.dot(self.h_layer.T, error)
        d_i_weights = np.dot(X.T, np.multiply(np.dot(error, self.o_weights.T), tanh_derivative(self.h_layer)))
        return d_o_weights, d_i_weights

    def backpropagation_desc(self, X, preds, error):
        d_o_weights, d_i_weights = self.backpropagation(X, error)
        self.i_weights -= self.alpha * (d_i_weights / preds.shape[0])
        self.o_weights -= self.alpha * (d_o_weights / preds.shape[0])
        return error

    def update_mse(self, preds, error):
        e_n = np.sum(np.multiply(error, error), axis=1) / 2
        self.mse = np.sum(e_n) / preds.shape[0]

    def next_batch(self, X, y, batchSize):
        # loop over our dataset `X` in mini-batches of size `batchSize`
        for i in np.arange(0, X.shape[0], batchSize):
            # yield a tuple of the current batched data and labels
            yield (X[i:i + batchSize], y[i:i + batchSize])

    def partial_fit(self, X, y):
        epoch_num = 0
        error_history = np.array([])
        X = np.c_[np.ones(len(X)), np.matrix(X)]
        y = np.matrix(y).T
        while self.mse > 10**-6 and epoch_num < self.epoch_max:
            epoch_error = np.array([])
            for (batch_x, batch_y) in self.next_batch(X, y, 32):
                # Training Phase
                preds = self.feedforward(batch_x)
                error = (preds - batch_y)
                self.backpropagation_desc(batch_x, preds, error)
                self.update_mse(preds, error)

                epoch_error = np.append(epoch_error, self.mse)
            error_history = np.append(error_history, np.average(epoch_error))
            epoch_num += 1
        return error_history, epoch_num


def main():
    (X, y) = make_blobs(n_samples=400, n_features=2, centers=2, cluster_std=2.5, random_state=95)
    hidden = 10
    output = 1
    alpha = 0.1
    mlp = MLP(2, hidden, 1, alpha, 100)
    train_error, epoches = mlp.partial_fit(X, y)
    plt.plot(range(0, epoches), train_error, label='Training MSE: ' + str(train_error[-1]))

    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
