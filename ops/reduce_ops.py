from deepen.backend import active_backend as bx

_bx = bx() # backend singleton

# Internal helper functions
def _normalize_axes(x, axes):
    if axes is None:
        return None
    if isinstance(axes, int):
        axes = (axes,)
    ndim = x.ndim
    return tuple(ax if ax >= 0 else ax + ndim for ax in axes)

def _count_elements(shape, axes):
    if axes is None:
        axes = range(len(shape))
    size = 1
    for ax in axes:
        size *= shape[ax]
    return size

# Summation
class sum_:
    @staticmethod
    def forward(save, x, axes=None):
        axes = _normalize_axes(x, axes)

        output = _bx.sum(x, axis=axes, keepdims=True)

        save.axes = axes
        save.x_shape = x.shape

        return output

    @staticmethod
    def backward(save, output_grad):
        dx = _bx.multiply(_bx.ones(save.x_shape, dtype=output_grad.dtype), output_grad)
        return dx,

# Mean (average)
class mean:
    @staticmethod
    def forward(save, x, axes=None):
        axes = _normalize_axes(x, axes)

        output = _bx.mean(x, axis=axes, keepdims=True)

        save.num_elements = _count_elements(x.shape, axes)
        save.x_shape = x.shape

        return output

    @staticmethod
    def backward(save, output_grad):
        factor = 1.0 / save.num_elements
        dx = _bx.multiply(_bx.ones(save.x_shape, dtype=output_grad.dtype), _bx.multiply(output_grad, factor))
        return dx,

# Minimum
class min_:
    @staticmethod
    def forward(save, x, axes=None):
        axes = _normalize_axes(x, axes)

        output = _bx.min(x, axis=axes, keepdims=True)

        mask = save.mask = _bx.equal(x, output) # True where value == min
        save.count = _bx.sum(mask, axis=axes, keepdims=True) # number of minima per bucket

        return output

    @staticmethod
    def backward(save, output_grad):
        dx = _bx.divide(_bx.multiply(output_grad, save.mask), save.count)
        return dx,

# Maximum
class max_:
    @staticmethod
    def forward(save, x, axes=None):
        axes = _normalize_axes(x, axes)

        output = _bx.max(x, axis=axes, keepdims=True)

        mask = save.mask = _bx.equal(x, output) # True where value == max
        save.count = _bx.sum(mask, axis=axes, keepdims=True) # number of maxima per bucket

        return output

    @staticmethod
    def backward(save, output_grad):
        dx = _bx.divide(_bx.multiply(output_grad, save.mask), save.count)
        return dx,

# Softmax
class softmax:
    @staticmethod
    def forward(save, x, axes=None):
        axes = _normalize_axes(x, axes)

        save.axes = axes

        # shift logits trick
        shift = _bx.max(x, axis=axes, keepdims=True)
        e = _bx.exp(x - shift)
        output = _bx.divide(e, _bx.sum(e, axis=axes, keepdims=True))

        save.output = output

        return output

    @staticmethod
    def backward(save, output_grad):
        dot = _bx.sum(_bx.multiply(output_grad, save.output), axis=save.axes, keepdims=True)
        dx = _bx.multiply(save.output, _bx.subtract(output_grad, dot))
        return dx,