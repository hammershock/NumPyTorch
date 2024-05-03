import pickle
import numpy as np


def exp(inp):
    return np.exp(inp)


def randn_like(inp):
    return np.random.randn(*inp.shape)


def unsqueeze(array, dim):
    return np.expand_dims(array, axis=dim)


def cat(inps, dim=0):
    return np.concatenate(inps, axis=dim)


def expand_copy(array, shape):
    """
    Expand the dimensions of a numpy array to the specified shape.
    Only dimensions with size 1 can be expanded.

    Parameters:
    - array (np.ndarray): The input array.
    - shape (tuple of int): The new shape expected. Dimensions must be compatible.

    Returns:
    - np.ndarray: A view of the array with expanded dimensions.
    """
    if not all((s == array.shape[i] or array.shape[i] == 1) for i, s in enumerate(shape)):
        raise ValueError("The new shape must be compatible with the original shape.")

    return np.broadcast_to(array, shape)


def load(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
        return data


def load_pytorch(model_path):
    import torch
    data = torch.load(model_path)
    return {f'{name}.value': value.cpu().numpy() for name, value in data.items()}


def save(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def FloatTensor(inp):
    # foo
    return inp


