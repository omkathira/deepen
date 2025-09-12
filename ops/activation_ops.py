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
        save.x = x
        output = _bx.tanh(x)
        save.output = output
        return output

    @staticmethod
    def backward(save, output_grad):
        dx = _bx.multiply(output_grad, _bx.subtract(_bx.ones_like(save.x), _bx.power(save.output, 2)))
        return dx,

# ReLU
class relu:
    @staticmethod
    def forward(save, x):
        save.x = x
        output = _bx.maximum(0, x)
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
        save.x = x
        slope = _bx.array(neg_slope, dtype=x.dtype)
        save.slope = slope
        output = _bx.maximum(_bx.multiply(x, slope), x)
        return output

    @staticmethod
    def backward(save, output_grad):
        grad_mask = _bx.where(save.x > 0, _bx.ones_like(save.x), save.slope)
        dx = _bx.multiply(output_grad, grad_mask)
        return dx,