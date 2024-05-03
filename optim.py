from abc import abstractmethod
from typing import List

import numpy as np

from nn import Parameter


class Optimizer:
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def zero_grad(self):
        pass

    @abstractmethod
    def step(self):
        pass


class SGD(Optimizer):
    def __init__(self, parameters: List[Parameter], lr, l2_regularization=0, l1_regularization=0):
        self.l2_regularization = l2_regularization
        self.l1_regularization = l1_regularization
        self.maintaining_params = parameters
        self.lr = lr

    def zero_grad(self):
        for param in self.maintaining_params:
            param.grad *= 0

    def step(self):
        for param in self.maintaining_params:
            param.value -= self.lr * param.grad
            if not param.is_bias:
                param.value -= self.lr * (2*self.l2_regularization*param.value + self.l1_regularization*np.sign(param.value))


class Adam(Optimizer):
    def __init__(self, parameters: List[Parameter], lr, beta1=0.9, beta2=0.999, epsilon=1e-8,
                 l2_regularization=0, l1_regularization=0):
        self.l2_regularization = l2_regularization
        self.l1_regularization = l1_regularization
        self.maintaining_params = parameters  # List to hold parameters to be optimized
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.t = 0  # Initialize the time step (iteration)
        # Initialize first moment estimate
        self.m = [0.] * len(self.maintaining_params)
        # Initialize second moment estimate
        self.v = [0.] * len(self.maintaining_params)

    def zero_grad(self):
        for param in self.maintaining_params:
            param.grad *= 0

    def step(self):
        self.t += 1
        for idx, param in enumerate(self.maintaining_params):

            param.value -= (self._calculate_delta(param, idx))

    def _calculate_delta(self, param, idx):
        if not param.is_bias:
            param.grad += 2*self.l2_regularization*param.value + self.l1_regularization*np.sign(param.value)
        self.m[idx] = self.beta1 * self.m[idx] + (1 - self.beta1) * param.grad  # Update first moment estimate
        self.v[idx] = self.beta2 * self.v[idx] + (1 - self.beta2) * (param.grad ** 2)  # Update second moment estimate
        # Bias-corrected first moment estimate
        m_hat = self.m[idx] / (1 - self.beta1 ** self.t)
        # Bias-corrected second moment estimate
        v_hat = self.v[idx] / (1 - self.beta2 ** self.t)
        # Calculate the parameter update
        return self.lr * m_hat / (v_hat ** 0.5 + self.epsilon)

