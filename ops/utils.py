from deepen.backend import active_backend as bx

_bx = bx() # backend singleton

# Broadcasting (for backward)
def _reduce_grad(grad, target_shape):
    if grad.shape == target_shape:
        return grad

    grad_ndim, target_ndim = len(grad.shape), len(target_shape)

    # leading axes case where grad shape is say (5, 3, 4) but we need (3, 4) so axis (0,) has to be reduced    
    leading = range(grad_ndim - target_ndim) # inner axes can never be missing

    # inner axes case where grad shape is say (5, 3, 4) but we need (1, 3, 4) so axis (0,) has to be reduced, or another
    # inner axes case where grad shape is say (2, 3, 4, 5) but we need (1, 3, 1, 5) so axes (0, 2) have to be reduced
    inner = [
        i + (grad_ndim - target_ndim)
        for i, (grad_dim, target_dim) in enumerate(zip(grad.shape[-target_ndim:], target_shape))
        if grad_dim != 1 and target_dim == 1
    ]

    axes = tuple(leading) + tuple(inner)

    reduced_grad = _bx.sum(grad, axis=axes, keepdims=True)

    return reduced_grad

# Internal helper functions
def _normalize_axes(x, axes):    
    if axes is None:
        return None
    
    if len(axes) != len(set(axes)):
        raise ValueError("recieved duplicate axes")
    
    if isinstance(axes, int):
        axes = (axes,)
        
    return tuple(ax if ax >= 0 else ax + x.ndim for ax in axes)

def _count_elements(shape, axes):
    if axes is None:
        axes = range(len(shape))

    size = 1
    for ax in axes:
        size *= shape[ax]

    return size

# Dropout
class dropout:
    @staticmethod
    def forward(save, x, p):
        q = save.q = 1.0 - p
        u = _bx.random.uniform(size=x.shape, low=0.0, high=1.0, dtype=x.dtype)
        mask = save.mask = _bx.less(u, q).astype(x.dtype)
        output = _bx.divide(_bx.multiply(x, mask), q)
        return output

    @staticmethod
    def backward(save, output_grad):
        dx = _bx.divide(_bx.multiply(output_grad, save.mask), save.q)
        return dx,

# Class to hold intermediate values needed for many ops
class Cache:
    # Used in most ops (based on relevance)
    x: object = None
    x_shape: object = None

    y: object = None
    y_shape: object = None

    # Only used in concatenate
    x_end: object = None
    y_end: object = None

    # Only used in pow
    n: object = None

    # Only used in log
    log_base_tensor: object = None

    # Only used in clip
    min_val: object = None
    max_val: object = None

    # Only used in reduction ops (based on relevance)
    axes: object = None
    mask: object = None
    out: object = None

    # Used in most activations (based on relevance)
    output: object = None

    # Only used in sigmoid
    ones: object = None

    # Only used in leaky_relu
    neg_slope: object = None

    # Only used in dropout
    q: object = None