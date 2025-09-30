from deepen.backend import active_backend as bx
from deepen.ops.utils import _reduce_grad # handles broadcasting for backward passes

_bx = bx() # backend singleton

# Inner product
class matmul:
    @staticmethod
    def forward(save, x, y):
        output = _bx.matmul(x, y)
        save.x, save.x_shape = x, x.shape
        save.y, save.y_shape = y, y.shape
        return output
    
    @staticmethod
    def backward(save, output_grad):
        dx = _reduce_grad(_bx.matmul(output_grad, save.y.T), save.x_shape)
        dy = _reduce_grad(_bx.matmul(save.x.T, output_grad), save.y_shape)
        return dx, dy

# Outer product
class outer:
    @staticmethod
    def forward(save, x, y):
        output = _bx.outer(x, y)
        save.x, save.x_shape = x, x.shape
        save.y, save.y_shape = y, y.shape
        return output
    
    @staticmethod
    def backward(save, output_grad):
        x_reshaped = _bx.reshape(save.x, (-1, 1))
        y_reshaped = _bx.reshape(save.y, (1, -1))
        dx = _reduce_grad(_bx.sum(_bx.multiply(output_grad, y_reshaped), axis=1), save.x_shape)
        dy = _reduce_grad(_bx.sum(_bx.multiply(output_grad, x_reshaped), axis=0), save.y_shape)
        return dx, dy