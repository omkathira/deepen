from deepen.backend import active_backend as bx
from deepen.ops.utils import _reduce_grad # handles broadcasting for backward passes

_bx = bx() # backend singleton

# Element-wise addition
class add:
    _save_data = ('x_shape', 'y_shape')
    
    @staticmethod
    def forward(save, x, y):
        output = _bx.add(x, y)

        if save.active:
            save.x_shape = x.shape
            save.y_shape = y.shape
        
        return output
    
    @staticmethod
    def backward(save, output_grad):
        dx = _reduce_grad(output_grad, save.x_shape)
        dy = _reduce_grad(output_grad, save.y_shape)
        return dx, dy

# Element-wise subtraction
class sub:
    _save_data = ('x_shape', 'y_shape')

    @staticmethod
    def forward(save, x, y):
        output = _bx.subtract(x, y)

        if save.active:
            save.x_shape = x.shape
            save.y_shape = y.shape
        
        return output
    
    @staticmethod
    def backward(save, output_grad):
        dx = _reduce_grad(output_grad, save.x_shape)
        dy = _reduce_grad(_bx.multiply(output_grad, -1), save.y_shape)
        return dx, dy
    
# Element-wise multiplication
class mul:
    _save_data = ('x', 'x_shape', 'y', 'y_shape')

    @staticmethod
    def forward(save, x, y):
        output = _bx.multiply(x, y)
        
        if save.active:
            save.x, save.x_shape = x, x.shape
            save.y, save.y_shape = y, y.shape

        return output
    
    @staticmethod
    def backward(save, output_grad):
        dx = _reduce_grad(_bx.multiply(output_grad, save.y), save.x_shape)
        dy = _reduce_grad(_bx.multiply(output_grad, save.x), save.y_shape)
        return dx, dy

# Element-wise division
class div:
    _save_data = ('x', 'x_shape', 'y', 'y_shape')

    @staticmethod
    def forward(save, x, y):
        output = _bx.divide(x, y)

        if save.active:
            save.x, save.x_shape = x, x.shape
            save.y, save.y_shape = y, y.shape
        
        return output
    
    @staticmethod
    def backward(save, output_grad):
        dx = _reduce_grad(_bx.divide(output_grad, save.y), save.x_shape)
        dy = _reduce_grad(_bx.divide(_bx.multiply(_bx.multiply(output_grad, -1), save.x), _bx.power(save.y, 2)), save.y_shape)
        return dx, dy

# Negation
class neg:
    _save_data = ('x_shape',)

    @staticmethod
    def forward(save, x):
        output = _bx.multiply(x, -1)

        if save.active:
            save.x_shape = x.shape

        return output
    
    @staticmethod
    def backward(save, output_grad):
        dx = _reduce_grad(_bx.multiply(output_grad, -1), save.x_shape)
        return dx,

# Absolute value
class abs_: 
    _save_data = ('x', 'x_shape')

    @staticmethod
    def forward(save, x):
        output = _bx.abs(x)

        if save.active:
            save.x, save.x_shape = x, x.shape
        
        return output
    
    @staticmethod
    def backward(save, output_grad):
        dx = _reduce_grad(_bx.multiply(output_grad, _bx.sign(save.x)), save.x_shape)
        return dx,

# Power (handles roots)
class pow_:
    _save_data = ('x', 'x_shape', 'n')

    @staticmethod
    def forward(save, x, n):
        output = _bx.power(x, n)

        if save.active:
            save.x, save.x_shape = x, x.shape
            save.n = n

        return output
    
    @staticmethod
    def backward(save, output_grad):
        dx = _reduce_grad(_bx.multiply(_bx.multiply(save.n, _bx.power(save.x, save.n - 1)), output_grad), save.x_shape)
        return dx,

# Exponentiation
class exp:
    _save_data = ('x_shape', 'output')

    @staticmethod
    def forward(save, x):
        output = _bx.exp(x)

        if save.active:
            save.x_shape = x.shape
            save.output = output
        
        return output
    
    @staticmethod
    def backward(save, output_grad):
        dx = _reduce_grad(_bx.multiply(output_grad, save.output), save.x_shape)
        return dx,

# Logarithm
class log:
    _save_data = ('x', 'x_shape', 'log_base')

    @staticmethod
    def forward(save, x, base=_bx.e):
        log_base = _bx.log(_bx.array(base, dtype=x.dtype))
        output = _bx.divide(_bx.log(x), log_base)

        if save.active:
            save.x, save.x_shape =  x, x.shape
            save.log_base = log_base

        return output
    
    @staticmethod
    def backward(save, output_grad):
        dx = _reduce_grad(_bx.multiply(output_grad, _bx.divide(1, _bx.multiply(save.x, save.log_base))), save.x_shape)
        return dx, None

# Clip
class clip:
    _save_data = ('x', 'x_shape', 'min_val', 'max_val')

    @staticmethod
    def forward(save, x, min_val, max_val):
        output = _bx.clip(x, min_val, max_val)

        if save.active:
            save.x, save.x_shape = x, x.shape
            save.min_val, save.max_val = min_val, max_val

        return output
    
    @staticmethod
    def backward(save, output_grad):
        mask = ((save.x >= save.min_val) & (save.x <= save.max_val)).astype(save.x.dtype)
        dx = _reduce_grad(_bx.multiply(output_grad, mask), save.x_shape)
        return dx, None, None