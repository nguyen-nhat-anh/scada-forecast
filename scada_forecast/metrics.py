import numpy as np


def ape(true, pred):
    return np.abs((true - pred) / true)

def accuracy(true, pred, threshold=0.02):
    ape_values = ape(true, pred)
    return np.sum(ape_values <= threshold) / len(ape_values)