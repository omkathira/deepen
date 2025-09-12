from deepen.backend import active_backend as bx
from deepen.ops.utils import reduce_grad # handles broadcasting for backward passes

_bx = bx() # backend singleton

# Element-wise addition
class add:
    @staticmethod
    def forward(save, x, y):
        save.x_shape, save.y_shape = x.shape, y.shape
        output = _bx.add(x, y)
        return output
    
    @staticmethod
    def backward(save, output_grad):
        dx = reduce_grad(output_grad, save.x_shape)
        dy = reduce_grad(output_grad, save.y_shape)
        return dx, dy

# Element-wise subtraction
class sub:
    @staticmethod
    def forward(save, x, y):
        save.x_shape, save.y_shape = x.shape, y.shape
        output = _bx.subtract(x, y)
        return output
    
    @staticmethod
    def backward(save, output_grad):
        dx = reduce_grad(output_grad, save.x_shape)
        dy = reduce_grad(_bx.multiply(output_grad, -1), save.y_shape)
        return dx, dy
    
# Element-wise multiplication
class mul:
    @staticmethod
    def forward(save, x, y):
        save.x, save.y = x, y
        save.x_shape, save.y_shape = x.shape, y.shape
        output = _bx.multiply(x, y)
        return output
    
    @staticmethod
    def backward(save, output_grad):
        dx = reduce_grad(_bx.multiply(output_grad, save.y), save.x_shape)
        dy = reduce_grad(_bx.multiply(output_grad, save.x), save.y_shape)
        return dx, dy

# Element-wise division
class div:
    @staticmethod
    def forward(save, x, y):
        save.x, save.y = x, y
        save.x_shape, save.y_shape = x.shape, y.shape
        output = _bx.divide(x, y)
        return output
    
    @staticmethod
    def backward(save, output_grad):
        dx = reduce_grad(_bx.divide(output_grad, save.y), save.x_shape)
        dy = reduce_grad(_bx.divide(_bx.multiply(_bx.multiply(output_grad, -1), save.x), _bx.pow(save.y, 2)), save.y_shape)
        return dx, dy

# Negation
class neg:
    @staticmethod
    def forward(save, x):
        save.x_shape = x.shape
        output = _bx.multiply(x, -1)
        return output
    
    @staticmethod
    def backward(save, output_grad):
        dx = reduce_grad(_bx.multiply(output_grad, -1), save.x_shape)
        return dx,

# Absolute value
class abs_: 
    @staticmethod
    def forward(save, x):
        save.x, save.x_shape = x, x.shape
        output = _bx.abs(x)
        return output
    
    @staticmethod
    def backward(save, output_grad):
        dx = reduce_grad(_bx.multiply(output_grad, _bx.sign(save.x)), save.x_shape)
        return dx,

# Power (handles roots)
class pow_:
    @staticmethod
    def forward(save, x, n):
        save.x, save.x_shape = x, x.shape
        save.n = n
        output = _bx.power(x, n)
        return output
    
    @staticmethod
    def backward(save, output_grad):
        dx = reduce_grad(_bx.multiply(_bx.multiply(save.n, _bx.power(save.x, save.n - 1)), output_grad), save.x_shape)
        return dx,

# Exponentiation
class exp:
    @staticmethod
    def forward(save, x):
        save.x_shape = x.shape
        output = _bx.exp(x)
        save.output = output
        return output
    
    @staticmethod
    def backward(save, output_grad):
        dx = reduce_grad(_bx.multiply(output_grad, save.output), save.x_shape)
        return dx,

# Logarithm
class log:
    @staticmethod
    def forward(save, x, base=_bx.e):
        save.x, save.x_shape = x, x.shape
        base_tensor = _bx.array(base, dtype=x.dtype)
        log_base_tensor = _bx.log(base_tensor)
        save.log_base_tensor = log_base_tensor
        output = _bx.divide(_bx.log(x), log_base_tensor)
        return output
    
    @staticmethod
    def backward(save, output_grad):
        dx = reduce_grad(_bx.multiply(output_grad, _bx.divide(1, _bx.multiply(save.x, save.log_base_tensor))), save.x_shape)
        return dx, None

# Clip
class clip:
    @staticmethod
    def forward(save, x, min_val, max_val):
        save.x, save.x_shape = x, x.shape
        save.min_val, save.max_val = min_val, max_val
        output = _bx.clip(x, min_val, max_val)
        return output
    
    @staticmethod
    def backward(save, output_grad):
        mask = (save.x >= save.min_val) & (save.x <= save.max_val)
        dx = reduce_grad(_bx.multiply(output_grad, mask.astype(save.x.dtype)), save.x_shape) # preserve dtype when applying mask
        return dx, None, None