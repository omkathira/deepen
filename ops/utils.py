from deepen.backend import active_backend as bx

_bx = bx() # backend singleton

# Broadcasting (for backward)
def _reduce_grad(grad, target_shape):
    if grad.shape == target_shape:
        return grad

    grad_ndim, target_ndim = len(grad.shape), len(target_shape)

    # leading axes case where grad shape is, say, (5, 3, 4) but we need (3, 4) so axis (0,) has to be reduced    
    leading = range(grad_ndim - target_ndim) # inner axes can never be missing

    # inner axes case where grad shape is, say, (5, 3, 4) but we need (1, 3, 4) so axis (0,) has to be reduced, or another
    # inner axes case where grad shape is, say, (2, 3, 4, 5) but we need (1, 3, 1, 5) so axes (0, 2) have to be reduced
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
    
    if isinstance(axes, int):
        axes = (axes,)
    
    if len(axes) != len(set(axes)):
        raise ValueError("recieved duplicate axes")
        
    return tuple(ax if ax >= 0 else ax + x.ndim for ax in axes)

def _count_elements(shape, axes):
    if axes is None:
        axes = range(len(shape))

    size = 1
    for ax in axes:
        size *= shape[ax]

    return size

def _make_cache(op_cls, active):
    fields = getattr(op_cls, '_save_data')
    cache = type(f'{op_cls.__name__}_cache', (), {'__slots__': ('active',) + fields})
    save = cache()
    save.active = active
    return save

# Class to control Tensor immutability in functional mode
class ImmutableData:
    def __init__(self, arr):
        self._arr = arr

    @property
    def size(self):
        return self._arr.size

    @property
    def shape(self):
        return self._arr.shape

    @property
    def ndim(self):
        return self._arr.ndim
        
    @property
    def dtype(self):
        return self._arr.dtype
    
    def __getitem__(self, key):
        return self._arr[key]
    
    def __setitem__(self, key, value):
        raise ValueError("cannot modify .data of an immutable Tensor")
    
    def __array__(self):
        return self._arr.copy()