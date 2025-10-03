from deepen.backend import active_backend as bx
from deepen.ops.utils import _normalize_axes, _count_elements

_bx = bx() # backend singleton

# Summation
class sum_:
    @staticmethod
    def forward(save, x, axes=None):
        axes = _normalize_axes(x, axes)
        output = _bx.sum(x, axis=axes, keepdims=True)

        if save.active:
            save.x_shape = x.shape

        return output

    @staticmethod
    def backward(save, output_grad):
        dx = _bx.broadcast_to(output_grad, save.x_shape)
        return dx,

# Mean (average)
class mean:
    @staticmethod
    def forward(save, x, axes=None):
        axes = _normalize_axes(x, axes)
        output = _bx.mean(x, axis=axes, keepdims=True)

        if save.active:
            save.x_shape = x.shape
            save.axes = axes

        return output

    @staticmethod
    def backward(save, output_grad):
        num_elements = _count_elements(save.x_shape, save.axes)
        factor = _bx.divide(1.0, num_elements, dtype=output_grad.dtype)
        dx = _bx.broadcast_to(_bx.multiply(output_grad, factor), save.x_shape)
        return dx,

# Minimum
class min_:
    @staticmethod
    def forward(save, x, axes=None):
        axes = _normalize_axes(x, axes)
        output = _bx.min(x, axis=axes, keepdims=True)

        if save.active:
            save.mask = _bx.equal(x, output)
            save.axes = axes # true where value == min

        return output

    @staticmethod
    def backward(save, output_grad):
        count = _bx.sum(save.mask, axis=save.axes, keepdims=True) # number of minima per bucket
        dx = _bx.divide(_bx.multiply(output_grad, save.mask), count)
        return dx,

# Maximum
class max_:
    @staticmethod
    def forward(save, x, axes=None):
        axes = _normalize_axes(x, axes)
        output = _bx.max(x, axis=axes, keepdims=True)

        if save.active:
            save.mask = _bx.equal(x, output)
            save.axes = axes # true where value == max

        return output

    @staticmethod
    def backward(save, output_grad):
        count = _bx.sum(save.mask, axis=save.axes, keepdims=True) # number of maxima per bucket
        dx = _bx.divide(_bx.multiply(output_grad, save.mask), count)
        return dx,

# Softmax
class softmax:
    @staticmethod
    def forward(save, x, axes=None):
        axes = _normalize_axes(x, axes)
        shifted_logits = _bx.subtract(x, _bx.max(x, axis=axes, keepdims=True)) # shift logits trick
        exp_logits = _bx.exp(shifted_logits)
        output = _bx.divide(exp_logits, _bx.sum(exp_logits, axis=axes, keepdims=True))

        if save.active:
            save.axes = axes
            save.output = output

        return output

    @staticmethod
    def backward(save, output_grad):
        dot = _bx.sum(_bx.multiply(output_grad, save.output), axis=save.axes, keepdims=True)
        dx = _bx.multiply(save.output, _bx.subtract(output_grad, dot))
        return dx,