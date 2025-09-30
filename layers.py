from deepen.backend import active_backend as bx
from deepen.core.tensor import Tensor, Parameter
from deepen.ops.stochastic_ops import *

_bx = bx() # backend singleton

class Layer:
    def __init__(self):
        self._parameters = dict()
        self._layers = dict()
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def __setattr__(self, name, value): 
        if isinstance(value, Layer): # register sublayers
            self._layers[name] = value
        elif isinstance(value, Parameter): # register parameters
            self._parameters[name] = value

        super().__setattr__(name, value)

    def parameters(self):
        params = list(self._parameters.values())
        for layer in self._layers.values():
            params.extend(layer.parameters())
        return params

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

class Quadratic(Layer):
    def __init__(self, in_feat, out_feat, weight_init="uniform", bias=True, bias_init="zeros"):
        super().__init__()
        weight_init_fn = getattr(Parameter, weight_init, None)
        if not callable(weight_init_fn):
            raise ValueError(f"unknown weight initializer")
        
        self.quad_weights = weight_init_fn((in_feat, in_feat))
        self.lin_weights = weight_init_fn((in_feat, out_feat))

        bias_init_fn = getattr(Parameter, bias_init, None)
        if not callable(bias_init_fn):
            raise ValueError(f"unknown bias initializer")
        
        self.bias = bias_init_fn((1, out_feat)) if bias else None

    def forward(self, t):
        quad_output = (t.matmul(self.quad_weights) * t).sum(axes=1).reshape(-1, 1)
        lin_output = t.matmul(self.lin_weights)
        output = quad_output + lin_output
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
    pass

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
    pass



# soon: convolutional layers (1d, 2d), pooling layers (max/avg, 1d, 2d), RNN layers (LSTM, GRU, orthogonal weight init), attention (need to add elu, gelu, more stochastic ops)

# class ODESolver(Layer):
#     def __init__(self, func: Layer, step_size=1e-2, steps=None, method="euler"):
#         super().__init__()
#         self.func = func
#         self.step_size = step_size
#         self.steps = steps
#         self.method = method
    
#     def forward(self, y0, t0=0.0, t1=1.0):
#         h = self.step_size
#         if self.steps is None:
#             n_steps = int(max(1, round((t1 - t0) / h)))
#         else:
#             n_steps = self.steps
#             h = (t1 - t0) / max(1, n_steps)

#         y = y0
#         t = t0

#         if self.method == "euler":
#             for _ in range(n_steps):
#                 dy = self.func(t, y)
#                 y = y + h * dy
#                 t = t + h
#             return y
#         elif self.method == "rk4":
#             for _ in range(n_steps):
#                 k1 = self.func(t, y)
#                 k2 = self.func(t + 0.5 * h, y + 0.5 * h * k1)
#                 k3 = self.func(t + 0.5 * h, y + 0.5 * h * k2)
#                 k4 = self.func(t + h, y + h * k3)
#                 y = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
#                 t = t + h
#             return y
#         else:
#             raise ValueError("unknown integration method")