from ast import Pass
from deepen.backend import active_backend as bx

_bx = bx() # backend singleton

# Sigmoid
class sigmoid:
    @staticmethod
    def forward(save, x):
        ones = _bx.ones_like(x)
        output = _bx.divide(ones, _bx.add(ones, _bx.exp(_bx.multiply(x, -1))))
        save.ones, save.output = ones, output
        return output

    @staticmethod
    def backward(save, output_grad):
        dx = _bx.multiply(output_grad, _bx.multiply(save.output, _bx.subtract(save.ones, save.output)))
        return dx,

# Tanh
class tanh:
    @staticmethod
    def forward(save, x):
        output = _bx.tanh(x)
        save.x, save.output = x, output
        return output

    @staticmethod
    def backward(save, output_grad):
        dx = _bx.multiply(output_grad, _bx.subtract(_bx.ones_like(save.x), _bx.power(save.output, 2)))
        return dx,

# ReLU
class relu:
    @staticmethod
    def forward(save, x):
        output = _bx.maximum(0, x)
        save.x = x
        return output

    @staticmethod
    def backward(save, output_grad):
        grad_mask = save.x > 0
        dx = _bx.multiply(output_grad, grad_mask)
        return dx,

# Leaky ReLU
class leaky_relu:
    @staticmethod
    def forward(save, x, neg_slope=0.1):
        slope = _bx.array(neg_slope, dtype=x.dtype)
        output = _bx.maximum(_bx.multiply(x, slope), x)
        save.x, save.slope = x, slope
        return output

    @staticmethod
    def backward(save, output_grad):
        grad_mask = _bx.where(save.x > 0, _bx.ones_like(save.x), save.slope)
        dx = _bx.multiply(output_grad, grad_mask)
        return dx,

# GELU
class gelu:
    pass

# Swish
class swish:
    @staticmethod
    def forward(save, x):
        ones = _bx.ones_like(x)
        sigmoid = _bx.divide(ones, _bx.add(ones, _bx.exp(_bx.multiply(x, -1))))
        output = _bx.multiply(x, sigmoid)
        save.x, save.ones, save.sigmoid, save.output = x, ones, sigmoid, output
        return output

    @staticmethod
    def backward(save, output_grad):
        dx = _bx.multiply(output_grad, _bx.add(save.sigmoid, _bx.multiply(save.x, _bx.multiply(save.sigmoid, _bx.subtract(save.ones, save.sigmoid)))))
        return dx,