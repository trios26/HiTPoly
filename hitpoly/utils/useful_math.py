import numpy as np
import torch


def rmse(true, pred):
    return np.sqrt(np.power(pred - true, 2))


def mse(true, pred):
    return np.power(pred - true, 2).mean()


def mae(true, pred):
    return np.absolute(pred - true).mean()


def scaler(data, minimum, maximum):
    return data * (maximum - minimum) + minimum


def charge_scaler(data):
    arr = (data - data.min()) / (data - data.min()).sum()
    return arr - arr.mean()
