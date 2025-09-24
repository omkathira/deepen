from deepen.backend import active_backend as bx
from deepen.core.tensor import Tensor
from deepen.ops.utils import dropout

_bx = bx() # backend singleton

class Layer:
    def __init__(self):
        self._parameters = dict()
        self._layers = dict()
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def __setattr__(self, name, value): 
        if isinstance(value, Layer): # register layers, essentially registers layer l + 1 as a child of layer l
            self._layers[name] = value
        elif isinstance(value, Tensor) and value.requires_grad: # register parameters
            self._parameters[name] = value

        super().__setattr__(name, value)

    def parameters(self):
        params = list(self._parameters.values())
        for layer in self._layers.values():
            params.extend(layer.parameters())
        return params

class Linear(Layer):
    def __init__(self, in_feat, out_feat, bias=True):
        super().__init__()
        self.weights = Tensor.random((in_feat, out_feat), requires_grad=True)
        self.bias = Tensor.zeros((1, out_feat), requires_grad=True) if bias else None

    def forward(self, t):
        output = t.matmul(self.weights)
        return output + self.bias if self.bias is not None else output

class Quadratic(Layer):
    def __init__(self, in_feat, out_feat, bias=True):
        super().__init__()
        self.quad_weights = Tensor.random((in_feat, in_feat), requires_grad=True)
        self.lin_weights = Tensor.random((in_feat, out_feat), requires_grad=True)
        self.bias = Tensor.zeros((1, out_feat), requires_grad=True) if bias else None

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
        output = Tensor._from_op(dropout, t, p=self.p)
        return output

class BatchNorm(Layer):
    pass
        
class LayerNorm(Layer):
    pass

# soon: convolutional layers, pooling layers