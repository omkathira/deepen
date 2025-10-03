from deepen.backend import active_backend as bx

_bx = bx() # backend singleton

# Sigmoid
class sigmoid:
    _save_data = ('ones', 'output')

    @staticmethod
    def forward(save, x):
        ones = _bx.ones_like(x)
        output = _bx.divide(ones, _bx.add(ones, _bx.exp(_bx.multiply(x, -1))))

        if save.active:
            save.ones = ones
            save.output = output

        return output

    @staticmethod
    def backward(save, output_grad):
        dx = _bx.multiply(output_grad, _bx.multiply(save.output, _bx.subtract(save.ones, save.output)))
        return dx,

# Tanh
class tanh:
    _save_data = ('x', 'output')
    
    @staticmethod
    def forward(save, x):
        output = _bx.tanh(x)

        if save.active:
            save.x = x
            save.output = output
        
        return output

    @staticmethod
    def backward(save, output_grad):
        dx = _bx.multiply(output_grad, _bx.subtract(_bx.ones_like(save.x), _bx.power(save.output, 2)))
        return dx,

# ReLU
class relu:
    _save_data = ('x',)

    @staticmethod
    def forward(save, x):
        output = _bx.maximum(0, x)

        if save.active:
            save.x = x

        return output

    @staticmethod
    def backward(save, output_grad):
        grad_mask = save.x > 0
        dx = _bx.multiply(output_grad, grad_mask)
        return dx,

# Leaky ReLU
class leaky_relu:
    _save_data = ('x', 'slope')

    @staticmethod
    def forward(save, x, neg_slope=0.1):
        slope = _bx.array(neg_slope, dtype=x.dtype)
        output = _bx.maximum(_bx.multiply(x, slope), x)

        if save.active:
            save.x = x
            save.slope = slope

        return output

    @staticmethod
    def backward(save, output_grad):
        grad_mask = _bx.where(save.x > 0, _bx.ones_like(save.x), save.slope)
        dx = _bx.multiply(output_grad, grad_mask)
        return dx,

# Swish
class swish:
    _save_data = ('x', 'ones', 'sigmoid', 'output')

    @staticmethod
    def forward(save, x):
        ones = _bx.ones_like(x)
        sigmoid = _bx.divide(ones, _bx.add(ones, _bx.exp(_bx.multiply(x, -1))))
        output = _bx.multiply(x, sigmoid)

        if save.active:
            save.x = x
            save.ones = ones
            save.sigmoid = sigmoid
            save.output = output

        return output

    @staticmethod
    def backward(save, output_grad):
        dx = _bx.multiply(output_grad, _bx.add(save.sigmoid, _bx.multiply(save.x, _bx.multiply(save.sigmoid, _bx.subtract(save.ones, save.sigmoid)))))
        return dx,