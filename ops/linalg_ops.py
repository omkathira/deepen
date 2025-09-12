from deepen.backend import active_backend as bx
from deepen.ops.utils import reduce_grad # handles broadcasting for backward passes

_bx = bx() # backend singleton

class matmul:
    @staticmethod
    def forward(save, x, y):
        save.x, save.x_shape = x, x.shape
        save.y, save.y_shape = y, y.shape
        output = _bx.matmul(x, y)
        return output
    
    @staticmethod
    def backward(save, output_grad):
        dx = reduce_grad(_bx.matmul(output_grad, save.y.T), save.x_shape)
        dy = reduce_grad(_bx.matmul(save.x.T, output_grad), save.y_shape)
        return dx, dy