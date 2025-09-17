from deepen.core.tensor import Tensor

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
    pass

class Dropout(Layer):
    pass

# to-do: normalization layers, convolutional layers, pooling layers