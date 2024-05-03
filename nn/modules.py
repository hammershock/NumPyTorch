from abc import abstractmethod
from typing import Mapping

import numpy as np

from .utils import one_hot_encode


class Parameter:
    def __init__(self, input_array, is_bias=False):
        self.value = input_array
        self.grad = np.zeros_like(input_array)
        self.is_bias = is_bias


class Module:
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def forward(self, inp: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, grad) -> np.ndarray:
        pass

    def __call__(self, *args, **kwargs) -> np.ndarray:
        return self.forward(*args, **kwargs)

    def state_dict(self) -> Mapping[str, np.ndarray]:
        result = {}
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, Module):
                for key, value in attr.state_dict().items():
                    result[f'{attr_name}.{key}'] = value
            elif isinstance(attr, Parameter):
                result[attr_name + '.value'] = attr.value
                result[attr_name + '.grad'] = attr.grad
        return result

    def load_state_dict(self, state_dict: Mapping[str, np.ndarray]):
        for key, value in state_dict.items():
            res = key.split('.')
            if len(res) == 2:
                param = getattr(self, res[0])
                setattr(param, res[1], value)
            elif len(res) > 2:
                submodule = getattr(self, res[0])
                submodule.load_state_dict({'.'.join(res[1:]): value})


class Linear(Module):
    """
        Implements a fully connected layer with NumPy.
        >>> import torch
        >>> import torch.nn as nn
        >>> weights = np.random.randn(3, 5)  # Random weights
        >>> bias = np.random.randn(3)
        >>> inputs = np.random.rand(1, 5)  # Example input tensor with 1 batch size and 5 features
        >>> linear = Linear(5, 3)
        >>> # Manually setting weights and biases to a known value for testing
        >>> linear.weight = Parameter(weights)
        >>> linear.bias = Parameter(bias)
        >>> output = linear.forward(inputs)
        >>> # Setup the corresponding PyTorch linear layer
        >>> torch_linear = nn.Linear(5, 3)
        >>> torch_linear.weight = nn.Parameter(torch.tensor(weights, dtype=torch.float32))
        >>> torch_linear.bias = nn.Parameter(torch.tensor(bias, dtype=torch.float32))
        >>> torch_output = torch_linear(torch.tensor(inputs, dtype=torch.float32))
        >>> np.allclose(output, torch_output.detach().numpy(), atol=1e-6)
        True
        """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # 权重和偏置初始化
        self.weight = Parameter(np.random.randn(out_features, in_features) / np.sqrt(in_features / 2))
        self.bias = Parameter(np.random.randn(out_features) / np.sqrt(in_features / 2), is_bias=True)
        self.prev_inp = None

    def forward(self, inp):
        # Store input for backward pass
        self.prev_inp = inp
        # Compute output
        return inp.dot(self.weight.value.T) + self.bias.value

    def backward(self, grad):
        # Compute gradients
        grad_input = grad.dot(self.weight.value)
        grad_weight = grad.T.dot(self.prev_inp)
        grad_bias = np.sum(grad, axis=0)

        # Update gradients in Parameter instances
        self.weight.grad += grad_weight
        self.bias.grad += grad_bias

        return grad_input


class SoftmaxLayer(Module):
    def __init__(self):
        self.last_output = None

    def forward(self, inp):
        """

		:param inp: (batch, num_features, 1)
		:return: output: (batch, num_features, 1)
		"""
        exps = np.exp(inp)
        self.last_output = exps / np.sum(exps, axis=1, keepdims=True)
        return self.last_output

    def backward(self, grad):  # grad应与输入inp有相同的形状
        """
		backward of softmax function
		:param grad: (batch, num_features, 1)
		:return: (batch, num_features, 1)
		"""

        input_grad = np.zeros_like(grad)
        for i in range(grad.shape[0]):
            s = self.last_output[i].squeeze(-1)
            g = grad[i].squeeze(-1)
            jacobian_mat = np.diagflat(s) - np.outer(s, s)
            input_grad[i] = np.dot(jacobian_mat, g).reshape(-1, 1)
        return input_grad


class ReLU(Module):
    def __init__(self):
        self.prev_inp = None

    def forward(self, inp):
        self.prev_inp = inp
        return np.maximum(inp, 0)

    def backward(self, grad):
        return grad * (self.prev_inp >= 0)  # prev_inp < 0的，回传的梯度直接为0，dead relu，可能导致梯度消失


class Sigmoid(Module):
    def __init__(self):
        self.prev_inp = None

    def forward(self, inp):
        self.prev_inp = inp
        return 1 / (1 + np.exp(-inp))

    def backward(self, grad):
        sigmoid_output = self.forward(self.prev_inp)
        return grad * sigmoid_output * (1 - sigmoid_output)


class LeakyReLULayer(Module):
    def __init__(self, leak=0.01):
        self.prev_inp = None
        self.leak = leak

    def forward(self, inp):
        self.prev_inp = inp
        return np.maximum(self.leak * inp, inp)

    def backward(self, grad):
        self.prev_inp[self.prev_inp < 0] = self.leak
        self.prev_inp[self.prev_inp >= 0] = 1
        return grad * self.prev_inp


class Sequential(Module):
    def __init__(self, *args):
        self.layers = list(args)
        self.if_train = True

    def train(self):
        self.if_train = True

    def eval(self):
        self.if_train = False

    def parameters(self):
        parameters = []
        for layer in self.layers:
            for param in vars(layer).values():
                if isinstance(param, Parameter):
                    parameters.append(param)
        return parameters

    def forward(self, inp) -> np.ndarray:
        for layer in self.layers:
            if isinstance(layer, DropoutLayer) or isinstance(layer, BatchNormLayer):
                inp = layer(inp, train=self.if_train)
            else:
                inp = layer(inp)
        return inp

    def backward(self, grad) -> np.ndarray:
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def state_dict(self) -> Mapping[str, np.ndarray]:
        result = {}
        for idx, layer in enumerate(self.layers):
            if isinstance(layer, Module):
                for key, value in layer.state_dict().items():
                    result[f'layers.{idx}.{key}'] = value
        return result

    def load_state_dict(self, state_dict: Mapping[str, np.ndarray]):
        for name, value in state_dict.items():
            parts = name.split('.')
            submodule = self.layers[int(parts[0])]
            submodule.load_state_dict({".".join(parts[1:]): value})


class SoftmaxRegression(Sequential):
    def __init__(self, in_features, out_features):
        super().__init__(Linear(in_features, out_features), SoftmaxLayer())


class DropoutLayer(Module):
    def __init__(self, dropout_rate=0.3):
        self.dropout_rate = dropout_rate
        self.filter = None
        self.if_train = True

    def forward(self, inp, train=True):
        self.if_train = train
        if train:
            # Generate a binary mask with dropout probability
            self.filter = np.random.binomial(n=1, p=1 - self.dropout_rate, size=inp.shape)
            return inp * self.filter
        return inp

    def backward(self, grad):
        # 对流过的梯度进行过滤，仅通过(1-p)的梯度流
        return grad * self.filter if self.if_train else grad


def get_im2col_indices(x_shape, field_height, field_width, padding=(1, 1), stride=(1, 1)):
    N, C, H, W = x_shape
    pad_h, pad_w = padding
    stride_h, stride_w = stride

    # Calculate output spatial dimensions
    out_height = (H + 2 * pad_h - field_height) // stride_h + 1
    out_width = (W + 2 * pad_w - field_width) // stride_w + 1

    # Generate offset grid
    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride_h * np.repeat(np.arange(out_height), out_width)

    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride_w * np.tile(np.arange(out_width), out_height)

    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

    return k.astype(int), i.astype(int), j.astype(int)


def im2col_indices(x, field_height, field_width, padding=(1, 1), stride=(1, 1)):
    """ Transform input image x into columns using specified field (kernel) dimensions, padding, and stride. """
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    p_h, p_w = padding  # Unpack padding tuple
    s_h, s_w = stride   # Unpack stride tuple

    # Add padding around the input
    x_padded = np.pad(x, ((0, 0), (0, 0), (p_h, p_h), (p_w, p_w)), mode='constant')

    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding, stride)

    # Extract patches from padded input using fancy indexing
    cols = x_padded[:, k, i, j]
    C = x.shape[1]  # Number of channels

    # Reshape extracted patches into columns
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols


def col2im_indices(cols, x_shape, field_height, field_width, padding=(1, 1), stride=(1, 1)):
    N, C, H, W = x_shape
    pad_h, pad_w = padding
    stride_h, stride_w = stride

    H_padded, W_padded = H + 2 * pad_h, W + 2 * pad_w
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)

    k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding, stride)

    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)  # Reshape cols back into the image shape

    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)

    # Trim padding to get the original x dimensions back
    if pad_h or pad_w:
        return x_padded[:, :, pad_h:-pad_h, pad_w:-pad_w]
    return x_padded


class Conv2d(Module):
    """
    Implements a convolutional layer with NumPy.
    >>> import torch
    >>> import torch.nn as nn
    >>> inputs = np.random.rand(1, 1, 4, 4)  # Batch size = 1, Channels = 1, Height = 4, Width = 4
    >>> weights = np.random.randn(1, 1, 3, 3)  # Random weights
    >>> bias = np.random.randn(1)  # Random bias
    >>> conv = Conv2d(1, 1, 3, stride=1, padding=1)
    >>> conv.weight = Parameter(weights)  # Set the same weights for NumPy implementation
    >>> conv.bias = Parameter(bias)  # Set the same bias for NumPy implementation
    >>> output = conv.forward(inputs)
    >>> torch_conv = nn.Conv2d(1, 1, 3, stride=1, padding=1)
    >>> torch_conv.weight = nn.Parameter(torch.tensor(weights, dtype=torch.float32))
    >>> torch_conv.bias = nn.Parameter(torch.tensor(bias, dtype=torch.float32))
    >>> torch_input = torch.tensor(inputs, dtype=torch.float32)
    >>> torch_output = torch_conv(torch_input)
    >>> np.allclose(output, torch_output.detach().numpy(), atol=1e-6)  # Check if the outputs are close enough
    True
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.weight = Parameter(np.random.randn(out_channels, in_channels, *self.kernel_size))
        self.bias = Parameter(np.random.randn(out_channels))

    def forward(self, inp):
        n_filters, d_filter, h_filter, w_filter = self.weight.value.shape
        n_x, d_x, h_x, w_x = inp.shape

        h_out = (h_x - h_filter + 2 * self.padding) // self.stride + 1
        w_out = (w_x - w_filter + 2 * self.padding) // self.stride + 1

        # Transform the input image into column shape
        x_col = im2col_indices(inp, h_filter, w_filter, padding=self.padding, stride=self.stride)
        # Reshape and permute the weights into a 2D array of shape (filter_height * filter_width * in_channels, out_channels)
        w_col = self.weight.value.reshape(n_filters, -1).T

        # Perform matrix multiplication between the reshaped weights and the image columns
        out = w_col.T @ x_col + self.bias.value.reshape(-1, 1)

        # Reshape the output into a suitable shape
        out = out.reshape(n_filters, h_out, w_out, n_x)
        out = out.transpose(3, 0, 1,
                            2)  # rearrange axis to match output dimensions as batch_size x num_filters x height x width

        return out

    def backward(self, dout) -> np.ndarray:
        raise NotImplementedError
        # dout: (batch, out_channels, out_h, out_d)
        din = np.zeros_like(self.last_input)

        # 3. 计算权重的梯度
        for i in range(self.output_height):
            for j in range(self.output_width):
                local_input = self.last_input[:, :, i:i + self.kernel_size, j:j + self.kernel_size]
                for k in range(self.out_channels):
                    self.weight.grad[k] += np.sum(dout[:, k, i, j][:, None, None, None] * local_input, axis=0)
                    self.bias.grad[k] += np.sum(dout[:, k, i, j])

        # 4. 计算输入的梯度
        # (batch, out_channels, out_h, out_d) -> (batch, in_channels, height, width)


class ConvTranspose2d(Module):
    """
        Implements a convolutional transpose layer with NumPy.

        >>> import torch
        >>> import torch.nn as nn
        >>> inputs = np.random.rand(1, 1, 4, 4)  # Example input tensor
        >>> weights = np.random.randn(1, 1, 3, 3)  # Random weights for a 3x3 kernel
        >>> bias = np.random.randn(1)  # Random bias
        >>> conv_transpose = ConvTranspose2d(1, 1, 3, stride=1, padding=1, output_padding=0)
        >>> conv_transpose.weight = Parameter(weights)  # Set the same weights for the NumPy implementation
        >>> conv_transpose.bias = Parameter(bias)  # Set the same bias for the NumPy implementation
        >>> output = conv_transpose.forward(inputs)
        >>> torch_conv_transpose = nn.ConvTranspose2d(1, 1, 3, stride=1, padding=1, output_padding=0)
        >>> torch_conv_transpose.weight = nn.Parameter(torch.tensor(weights, dtype=torch.float32))
        >>> torch_conv_transpose.bias = nn.Parameter(torch.tensor(bias, dtype=torch.float32))
        >>> torch_inputs = torch.tensor(inputs, dtype=torch.float32)
        >>> torch_output = torch_conv_transpose(torch_inputs)
        >>> np.allclose(output, torch_output.detach().numpy(), atol=1e-6)  # Check if the outputs are close enough
        True
        """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.output_padding = output_padding if isinstance(output_padding, tuple) else (output_padding, output_padding)

        # Initialize weights and biases
        self.weight = Parameter(np.random.randn(in_channels, out_channels, *self.kernel_size))
        self.bias = Parameter(np.random.randn(out_channels))

    def forward(self, inp):
        n, c, h, w = inp.shape
        d_w, d_h = self.weight.value.shape[2], self.weight.value.shape[3]
        out_h = (h - 1) * self.stride[0] - 2 * self.padding[0] + d_h + self.output_padding[0]
        out_w = (w - 1) * self.stride[1] - 2 * self.padding[1] + d_w + self.output_padding[1]

        # We need to col2im back to output dimensions
        inp_reshaped = inp.transpose(1, 0, 2, 3).reshape(c, -1)
        weights_reshaped = self.weight.value.reshape(self.in_channels, -1).T

        out_cols = weights_reshaped @ inp_reshaped
        out = col2im_indices(out_cols, (n, self.out_channels, out_h, out_w), d_h, d_w, self.padding, self.stride)

        out += self.bias.value.reshape(1, self.out_channels, 1, 1)

        return out

    def backward(self, dout) -> np.ndarray:
        raise NotImplementedError

class BatchNormLayer(Module):
    def __init__(self, n_features, eps=1e-5, momentum=0.9):
        self.eps = eps
        self.momentum = momentum
        self.n_features = n_features

        self.if_train = False

        # Learnable parameters for scaling and shifting
        self.gamma = Parameter(np.zeros(n_features))
        self.beta = Parameter(np.zeros(n_features))

        # Variables to track running statistics during training
        self.running_mean = 0
        self.running_var = 0

        self.out_ = None
        self.prev_inp = None
        self.sample_mean = None
        self.sample_var = None

    def forward(self, inp, train=True):
        inp = inp.squeeze(-1)  # (batch_size, num_features, 1) -> (batch_size, num_features)
        self.prev_inp = inp

        if train:
            # Compute sample mean and variance
            self.sample_mean = np.mean(inp, axis=0)
            self.sample_var = np.var(inp, axis=0)

            # Normalize the input using batch statistics
            self.out_ = (inp - self.sample_mean) / np.sqrt(self.sample_var + self.eps)

            # Update running statistics using momentum
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.sample_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.sample_var

            # Scale and shift the normalized output
            out = self.gamma.value * self.out_ + self.beta.value
        else:
            # During inference, use running statistics to normalize the input
            scale = self.gamma.value / np.sqrt(self.running_var + self.eps)
            out = inp * scale + (self.beta.value - self.running_mean * scale)

        return out[..., np.newaxis]

    def backward(self, grad):
        grad = grad.squeeze(-1)

        N = self.prev_inp.shape[0]
        dout_ = self.gamma.value * grad

        # Compute gradients for backpropagation
        dvar = np.sum(dout_ * (self.prev_inp - self.sample_mean)
                      * -0.5 * (self.sample_var + self.eps) ** -1.5, axis=0)
        dx_ = 1 / np.sqrt(self.sample_var + self.eps)
        dvar_ = 2 * (self.prev_inp - self.sample_mean) / N

        di = dout_ * dx_ + dvar * dvar_
        dmean = -1 * np.sum(di, axis=0)
        dmean_ = np.ones_like(self.prev_inp) / N

        dx = di + dmean * dmean_
        self.gamma.grad = np.sum(grad * self.out_, axis=0)
        self.beta.grad = np.sum(grad, axis=0)

        return dx[..., np.newaxis]


class Flatten(Module):
    def __init__(self):
        self.out_d = None
        self.out_h = None
        self.out_channels = None
        self.batch = None

    def forward(self, inp: np.ndarray) -> np.ndarray:
        # (batch, out_channels, out_h, out_d) -> (batch, num_features, 1)
        self.batch, self.out_channels, self.out_h, self.out_d = inp.shape
        return inp.reshape((self.batch, -1, 1))

    def backward(self, grad) -> np.ndarray:
        # (batch, num_features, 1) -> (batch, out_channels, out_h, out_d)
        return grad.reshape((self.batch, self.out_channels, self.out_h, self.out_d))


class Unflatten(Module):
    """
    # Input tensor with shape (batch_size, num_features)
    >>> batch_size = 2
    >>> num_features = 12  # Suppose the output shape we target is (3, 4)
    >>> input_tensor = np.random.randn(batch_size, num_features)

    # Target output shape to unflatten second dimension into (3, 4)
    >>> unflatten = Unflatten(1, (3, 4))
    >>> output = unflatten(input_tensor)
    >>> output.shape
    (2, 3, 4)
    """
    def __init__(self, dim, unflatten_shape):
        super().__init__()
        self.dim = dim
        self.unflatten_shape = unflatten_shape

    def forward(self, inp: np.ndarray) -> np.ndarray:
        """
        Reshape the specified dimension of the input tensor to the target shape.
        :param inp: Input array of shape (batch_size, ..., num_features, ...)
        :return: Output array with the specified dimension replaced by the unflatten_shape.
        """
        shape = list(inp.shape)
        # Replace the dimension at self.dim with multiple dimensions from self.unflatten_shape
        shape = shape[:self.dim] + list(self.unflatten_shape) + shape[self.dim+1:]
        return inp.reshape(shape)

    def backward(self, grad) -> np.ndarray:
        """
        Reverse the forward reshape operation to flatten the tensor back to its original shape.
        :param grad: Gradient array of the output shape from the forward pass.
        :return: Gradient reshaped back to the input shape of the forward pass.
        """
        return grad.reshape(grad.shape[0], -1, 1)


class AdaptiveAvgPool2d(Module):
    """
    Implements an adaptive average pooling operation in NumPy.

    >>> import torch
    >>> import torch.nn as nn
    >>> np.random.seed(0)
    >>> inputs = np.random.rand(1, 2, 6, 6)  # Example input tensor with 1 batch, 2 channels, and 6x6 spatial dimensions
    >>> adaptive_pool = AdaptiveAvgPool2d((2, 2))  # Targeting a 2x2 output grid
    >>> output = adaptive_pool.forward(inputs)
    >>> torch_adaptive_pool = nn.AdaptiveAvgPool2d((2, 2))
    >>> torch_inputs = torch.tensor(inputs, dtype=torch.float32)
    >>> torch_output = torch_adaptive_pool(torch_inputs)
    >>> np.allclose(output, torch_output.detach().numpy(), atol=1e-6)
    True
    """
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size  # output_size is expected to be a tuple (output_height, output_width)

    def forward(self, inp):
        batch_size, channels, height, width = inp.shape
        output_height, output_width = self.output_size

        output = np.zeros((batch_size, channels, output_height, output_width))

        # Calculate size of each pooling region
        stride_h = height / output_height
        stride_w = width / output_width

        for i in range(output_height):
            for j in range(output_width):
                # Calculate the boundaries of the pooling region
                h_start = int(i * stride_h)
                h_end = int(np.ceil((i + 1) * stride_h))
                w_start = int(j * stride_w)
                w_end = int(np.ceil((j + 1) * stride_w))

                # Ensure the pooling region captures at least one pixel
                h_end = max(h_end, h_start + 1)
                w_end = max(w_end, w_start + 1)

                # Handle cases where dimensions might exceed input dimensions
                h_end = min(h_end, height)
                w_end = min(w_end, width)

                # Select the region and apply the average pooling
                patch = inp[:, :, h_start:h_end, w_start:w_end]
                output[:, :, i, j] = np.mean(patch, axis=(2, 3))

        return output

    def backward(self, grad) -> np.ndarray:
        raise NotImplementedError


class CrossEntropyLoss:
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @staticmethod
    def forward(logits: np.ndarray, targets, epsilon=1e-12):
        batch_size, num_classes, _ = logits.shape
        if len(targets.shape) == 1:  # (batch_size, )
            targets = one_hot_encode(targets, num_classes)

        s = np.exp(logits - np.max(logits, keepdims=True, axis=1))
        y_pred = s / np.sum(s, axis=1, keepdims=True)
        loss = - np.sum(targets * np.log(y_pred + epsilon)) / batch_size
        grad = y_pred - targets
        return loss, grad
