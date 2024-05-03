import numpy as np


def one_hot_encode(n, num_features) -> np.ndarray:
    batch_size = n.shape[0]
    one_hot_vector = np.zeros((batch_size, num_features, 1))
    for i, item in enumerate(n):
        one_hot_vector[i, item, :] = 1.
    return one_hot_vector
