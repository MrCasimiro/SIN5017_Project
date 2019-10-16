import numpy as np
import pandas as pd
import ipdb

def tanh_derivative(y):
    return 1 - np.multiply(y, y)

def normal_values(data):
    values = np.array(data.values)
    mean = np.mean(values)
    std = np.std(values)
    values = (values - mean) / std
    return values, mean, std

def recurrent_train_val_test(filename):
    data = pd.read_csv(filename, sep="\n", header=None)
    values, mean, std = normal_values(data)
    x = values[0:-2]
    target = values[1:-1]
    index = int(len(x) * 0.15)
    train_x = x[index:len(x) - index]
    train_y = target[index:len(target) - index]
    val_x = x[0:index]
    val_y = target[0:index]
    test_x = x[len(x) - index:len(x)+1]
    test_y = target[len(target) - index:len(target)+1]
    test_y = (test_y * std) + mean
    return train_x, train_y, val_x, val_y, test_x, test_y

def mlp_train_val_test(filename, L):
    data = pd.read_csv(filename, sep="\n", header=None)
    values, mean, std = normal_values(data)
    x = []
    target = []
    for index in range(0, len(values) - L - 1):
        x.append(values[index:index+L].flatten())
        target.append(values[index+L+1].flatten())
    x = np.matrix(x)
    target = np.matrix(target)
    index = int(len(x) * 0.15)
    train_x = x[index:len(x) - index]
    train_y = target[index:len(target) - index]
    val_x = x[0:index]
    val_y = target[0:index]
    test_x = x[len(x) - index:len(x)+1]
    test_y = target[len(target) - index:len(target)+1]
    test_y = (test_y * std) + mean
    return train_x, train_y, val_x, val_y, test_x, test_y
    