import numpy as np

from .utils import one_hot_encode


def mse_loss(pred, target):
    """
    J = 1/2 * (y - y_hat)^2
    :param pred: np.ndarray, predictions with the shape of (batch, num_features, 1)
    :param target: targets with the shape of (batch, num_features, 1)
    :return: loss (scalar), grad (batch, num_features, 1)
    """
    loss = np.mean((pred - target) ** 2)
    grad = pred - target
    return loss, grad


def cross_entropy_loss(preds: np.ndarray, targets: np.ndarray, epsilon=1e-12):
    """
	Computes cross entropy between targets (encoded as one-hot vectors)
	and predictions.
	Input: predictions (np.ndarray) of shape (batch_size, n_classes, 1)
		   targets (np.ndarray) of shape (batch_size, n_classes, 1)
	Returns: loss (scalar) - the cross entropy loss
			 grad (np.ndarray) - gradient of the loss w.r.t. predictions of shape (batch_size, n_classes, 1)
	"""
    # Reshape inputs
    if len(targets.shape) == 1:  # (batch_size, )
        num_classes = preds.shape[1]
        targets = one_hot_encode(targets, num_classes)

    preds = preds.squeeze(-1)  # (batch_size, n_classes, 1) -> (batch_size, n_classes)
    targets = targets.squeeze(-1)

    # Number of samples
    N = preds.shape[0]

    # Compute cross entropy loss
    ce_loss = -np.sum(targets * np.log(preds + epsilon)) / N

    # Gradient of cross entropy loss w.r.t. softmax input
    grad = -targets / (preds + epsilon)

    return ce_loss, grad[:, :, None]
