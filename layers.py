from deepen.backend import active_backend as bx
from deepen.core.tensor import Tensor, Parameter
from deepen.ops.linalg_ops import _im2col
from deepen.ops.stochastic_ops import *

_bx = bx() # backend singleton

class Layer:
    def __init__(self):
        self._layers = {}
        self._parameters = {}

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def __setattr__(self, name, value): 
        if isinstance(value, Layer): # register layers
            self._layers[name] = value
        elif isinstance(value, Parameter): # register parameters
            self._parameters[name] = value

        super().__setattr__(name, value)

    def parameters(self):
        for p in self._parameters.values():
            yield p
        for layer in self._layers.values():
            yield from layer.parameters()

class Linear(Layer):
    def __init__(self, in_feat, out_feat, weight_init="uniform", bias=True, bias_init="zeros"):
        super().__init__()
        weight_init_fn = getattr(Parameter, weight_init, None)
        if not callable(weight_init_fn):
            raise ValueError(f"unknown weight initializer")
        
        self.weights = weight_init_fn((in_feat, out_feat))

        bias_init_fn = getattr(Parameter, bias_init, None)
        if not callable(bias_init_fn):
            raise ValueError(f"unknown bias initializer")
        
        self.bias = bias_init_fn((1, out_feat)) if bias else None

    def forward(self, t):
        output = t.matmul(self.weights)
        return output + self.bias if self.bias is not None else output

class Dropout(Layer):
    def __init__(self, p, train=True):
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError("invalid dropout probability")
        self.p = p
        self.train = train
    
    def forward(self, t):
        if self.p == 0.0 or not self.train:
            return t
        output = Tensor._from_op(dropout, t, self.p)
        return output

class GaussianNoise(Layer):
    def __init__(self,  mean=0.0, std=0.1, train=True):
        super().__init__()
        self.mean = mean
        self.std = std
        self.train = train
    
    def forward(self, t):
        if self.std == 0.0 or not self.train:
            return t
        output = Tensor._from_op(gaussian_noise, t, self.mean, self.std)
        return output

class LayerNorm1d(Layer):
    def __init__(self, in_feat, bias=True, epsilon=1e-5):
        super().__init__()
        self.weights = Parameter.ones((1, in_feat))
        self.bias = Parameter.zeros((1, in_feat)) if bias else None
        self.epsilon = epsilon

    def forward(self, t):
        mean = t.mean(axes=1)
        var = ((t - mean) ** 2).mean(axes=1)
        norm = (t - mean) / (var + self.epsilon) ** 0.5
        output = self.weights * norm
        return output + self.bias if self.bias is not None else output

class LayerNorm2d(Layer):
    def __init__(self, in_channels, bias=True, epsilon=1e-5):
        super().__init__()
        self.weights = Parameter.ones((1, in_channels, 1, 1))
        self.bias = Parameter.zeros((1, in_channels, 1, 1)) if bias else None
        self.epsilon = epsilon

    def forward(self, t):
        mean = t.mean(axes=1)
        var = ((t - mean) ** 2).mean(axes=1)
        norm = (t - mean) / (var + self.epsilon) ** 0.5
        output = self.weights * norm
        return output + self.bias if self.bias is not None else output

class BatchNorm1d(Layer):
    def __init__(self, in_feat, bias=True, momentum=0.9, train=True, epsilon=1e-5):
        super().__init__()
        self.weights = Parameter.ones((1, in_feat))
        self.bias = Parameter.zeros((1, in_feat)) if bias else None
        self.momentum = momentum
        self.train = train
        self.epsilon = epsilon

        self.running_mean = Tensor.zeros(in_feat)
        self.running_var = Tensor.ones(in_feat)

    def forward(self, t):
        if self.train:
            mean = t.mean(axes=0)
            var = ((t - mean) ** 2).mean(axes=0)
            
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
            
            norm = (t - mean) / (var + self.epsilon) ** 0.5
        else:
            norm = (t - self.running_mean) / (self.running_var + self.epsilon) ** 0.5

        output = self.weights * norm
        return output + self.bias if self.bias is not None else output

class BatchNorm2d(Layer):
    def __init__(self, in_channels, bias=True, momentum=0.9, train=True, epsilon=1e-5):
        super().__init__()
        self.weights = Parameter.ones((1, in_channels, 1, 1))
        self.bias = Parameter.zeros((1, in_channels, 1, 1)) if bias else None
        self.momentum = momentum
        self.train = train
        self.epsilon = epsilon

        self.running_mean = Tensor.zeros((1, in_channels, 1, 1))
        self.running_var = Tensor.ones((1, in_channels, 1, 1))

    def forward(self, t):
        if self.train:
            mean = t.mean(axes=(0, 2, 3))
            var = ((t - mean) ** 2).mean(axes=(0, 2, 3))

            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var

            norm = (t - mean) / (var + self.epsilon) ** 0.5
        else:
            norm = (t - self.running_mean) / (self.running_var + self.epsilon) ** 0.5

        output = self.weights * norm
        return output + self.bias if self.bias is not None else output

class Conv2d(Layer):
    def __init__(self, input_shape, num_filters, kernel_size=(3, 3), stride=1, padding=1, weight_init="uniform", bias=True, bias_init="zeros"):
        super().__init__()
        self.C, self.H, self.W = input_shape
        self.num_filters = num_filters
        self.k_h, self.k_w = kernel_size
        self.stride = stride
        self.padding = padding

        weight_init_fn = getattr(Parameter, weight_init, None)
        if not callable(weight_init_fn):
            raise ValueError(f"unknown weight initializer")
        
        self.weights = weight_init_fn((num_filters, self.C, self.k_h, self.k_w))

        bias_init_fn = getattr(Parameter, bias_init, None)
        if not callable(bias_init_fn):
            raise ValueError(f"unknown bias initializer")
        
        self.bias = bias_init_fn((1, num_filters, 1, 1)) if bias else None
    
    def forward(self, t):
        N, _, H, W = t.shape

        H_out = (H + 2 * self.padding - self.k_h) // self.stride + 1
        W_out = (W + 2 * self.padding - self.k_w) // self.stride + 1

        im2col_output = Tensor._from_op(_im2col, t, k_h=self.k_h, k_w=self.k_w, stride=self.stride, padding=self.padding)
        W_flat = self.weights.reshape(self.num_filters, -1)

        output = W_flat.matmul(im2col_output)
        output = output.reshape(self.num_filters, H_out, W_out, N).transpose((3, 0, 1, 2))
        return output + self.bias if self.bias is not None else output

class MaxPool2d(Layer):
    pass

class Embedding(Layer):
    def __init__(self, vocab_size, latent_feat, weight_init="uniform"):
        super().__init__()
        self.vocab_size = vocab_size
        self.latent_feat = latent_feat

        weight_init_fn = getattr(Parameter, weight_init, None)
        if not callable(weight_init_fn):
            raise ValueError(f"unknown weight initializer")
        
        self.weights = weight_init_fn((vocab_size, latent_feat))
    
    def forward(self, t):
        return self.weights.gather(t)

class PositionalEncoding(Layer):
    def __init__(self, seq_len, latent_feat):
        super().__init__()
        self.seq_len = seq_len
        self.latent_feat = latent_feat

        

class MultiHeadAttention(Layer):
    def __init__(self, in_feat, latent_feat, num_heads=4):
        super().__init__()
        if latent_feat % num_heads != 0:
            raise ValueError("latent_feat must be divisible by num_heads")

        self.latent_feat = latent_feat
        self.num_heads = num_heads
        self.att_head_feat = latent_feat / num_heads

        self.q_proj = Linear(in_feat, latent_feat)
        self.k_proj = Linear(in_feat, latent_feat)
        self.v_proj = Linear(in_feat, latent_feat)

        self.out_proj = Linear(latent_feat, in_feat)

    def _split_heads(self, t):
        batch_size, seq_len, _ = t.shape
        t = t.reshape(batch_size, seq_len, self.num_heads, self.att_head_feat)

# soon: RNN layers (LSTM, GRU, orthogonal weight init), reminder: concatenate, gather