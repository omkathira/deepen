from deepen.backend import active_backend as bx
from deepen.ops.utils import _reduce_grad # handles broadcasting for backward passes

_bx = bx() # backend singleton

# Equal to
class eq:
    @staticmethod
    def forward(save, x, y):
        output = _bx.equal(x, y)
        save.x_shape, save.y_shape = x.shape, y.shape
        return output

    @staticmethod
    def backward(save, output_grad):
        dx = _reduce_grad(_bx.zeros(shape=save.x_shape, dtype=output_grad.dtype), save.x_shape)
        dy = _reduce_grad(_bx.zeros(shape=save.y_shape, dtype=output_grad.dtype), save.y_shape)
        return dx, dy

# Not equal to
class ne:
    @staticmethod
    def forward(save, x, y):
        output = _bx.not_equal(x, y)
        save.x_shape, save.y_shape = x.shape, y.shape
        return output

    @staticmethod
    def backward(save, output_grad):
        dx = _reduce_grad(_bx.zeros(shape=save.x_shape, dtype=output_grad.dtype), save.x_shape)
        dy = _reduce_grad(_bx.zeros(shape=save.y_shape, dtype=output_grad.dtype), save.y_shape)
        return dx, dy

# Less than
class lt:
    @staticmethod
    def forward(save, x, y):
        output = _bx.less(x, y)
        save.x_shape, save.y_shape = x.shape, y.shape
        return output

    @staticmethod
    def backward(save, output_grad):
        dx = _reduce_grad(_bx.zeros(shape=save.x_shape, dtype=output_grad.dtype), save.x_shape)
        dy = _reduce_grad(_bx.zeros(shape=save.y_shape, dtype=output_grad.dtype), save.y_shape)
        return dx, dy

# Less than or equal to
class le:
    @staticmethod
    def forward(save, x, y):
        output = _bx.less_equal(x, y)
        save.x_shape, save.y_shape = x.shape, y.shape
        return output

    @staticmethod
    def backward(save, output_grad):
        dx = _reduce_grad(_bx.zeros(shape=save.x_shape, dtype=output_grad.dtype), save.x_shape)
        dy = _reduce_grad(_bx.zeros(shape=save.y_shape, dtype=output_grad.dtype), save.y_shape)
        return dx, dy

# Greater than
class gt:
    @staticmethod
    def forward(save, x, y):
        output = _bx.greater(x, y)
        save.x_shape, save.y_shape = x.shape, y.shape
        return output

    @staticmethod
    def backward(save, output_grad):
        dx = _reduce_grad(_bx.zeros(shape=save.x_shape, dtype=output_grad.dtype), save.x_shape)
        dy = _reduce_grad(_bx.zeros(shape=save.y_shape, dtype=output_grad.dtype), save.y_shape)
        return dx, dy

# Greater than or equal to
class ge:
    @staticmethod
    def forward(save, x, y):
        output = _bx.greater_equal(x, y)
        save.x_shape, save.y_shape = x.shape, y.shape
        return output

    @staticmethod
    def backward(save, output_grad):
        dx = _reduce_grad(_bx.zeros(shape=save.x_shape, dtype=output_grad.dtype), save.x_shape)
        dy = _reduce_grad(_bx.zeros(shape=save.y_shape, dtype=output_grad.dtype), save.y_shape)
        return dx, dy

# Not
class not_:
    @staticmethod
    def forward(save, x):
        output = _bx.logical_not(x)
        save.x_shape = x.shape
        return output

    @staticmethod
    def backward(save, output_grad):
        dx = _reduce_grad(_bx.zeros(shape=save.x_shape, dtype=output_grad.dtype), save.x_shape)
        return dx,

# And
class and_:
    @staticmethod
    def forward(save, x, y):
        output = _bx.logical_and(x, y)
        save.x_shape, save.y_shape = x.shape, y.shape
        return output

    @staticmethod
    def backward(save, output_grad):
        dx = _reduce_grad(_bx.zeros(shape=save.x_shape, dtype=output_grad.dtype), save.x_shape)
        dy = _reduce_grad(_bx.zeros(shape=save.y_shape, dtype=output_grad.dtype), save.y_shape)
        return dx, dy

# Or
class or_:
    @staticmethod
    def forward(save, x, y):
        output = _bx.logical_or(x, y)
        save.x_shape, save.y_shape = x.shape, y.shape
        return output

    @staticmethod
    def backward(save, output_grad):
        dx = _reduce_grad(_bx.zeros(shape=save.x_shape, dtype=output_grad.dtype), save.x_shape)
        dy = _reduce_grad(_bx.zeros(shape=save.y_shape, dtype=output_grad.dtype), save.y_shape)
        return dx, dy