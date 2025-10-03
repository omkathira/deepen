from deepen.backend import active_backend as bx
from deepen.core.tensor import Tensor
from deepen.ops.ewise_ops import *
from deepen.ops.logical_ops import *
from deepen.ops.index_ops import *
from deepen.ops.shape_ops import *
from deepen.ops.reduction_ops import *
from deepen.ops.linalg_ops import *
from deepen.ops.activation_ops import *
from deepen.ops.utils import Cache

_bx = bx() # backend singleton

_TRACE_DEPTH = 0

def _is_tracing():
    return _TRACE_DEPTH > 0

class Fensor:
    def __init__(self, data, requires_grad=False):
        data = _bx.array(data)

        object.__setattr__(self, "_Tensor", Tensor(data, requires_grad=requires_grad))
        
    def __setattr__(self, name, value):
        raise AttributeError("Fensor is immutable")
    
    @property
    def data(self):
        class ReadOnlyArray:
            def __init__(self, arr):
                self._arr = arr
            
            def __getitem__(self, key):
                return self._arr[key]
            
            def __setitem__(self, key, value):
                raise ValueError("cannot modify .data - Fensor is immutable")
            
            def __array__(self):
                return self._arr.copy()  # Return copy for safety
            
            def __repr__(self):
                return repr(self._arr)
        
        return ReadOnlyArray(self._Tensor.data)
    
    @property
    def requires_grad(self):
        return self._Tensor.requires_grad
    
    @property
    def shape(self):
        return self._Tensor.shape
    
    def detach(self):
        return Fensor(self.data, requires_grad=False)

def _any_traced_Fensor(*items):
    for item in items:
        if isinstance(item, Fensor) and item._Tensor.data is None:
            return True
    return False

def _to_Tensor(x, requires_grad):
    return Tensor(x._Tensor.data, requires_grad=requires_grad)

def _compute_eager(op_cls, *args, **kwargs):
    args = tuple(
        _to_Tensor(t, False) if isinstance(t, Fensor) else t
        for t in args
    )

    kwargs = {
        k: (_to_Tensor(v, False) if isinstance(v, Fensor) else v)
        for k, v in kwargs.items()
    }

    save = Cache(active=False)
    data = op_cls.forward(save, *args, **kwargs)
    output = Fensor(data, requires_grad=False)

    return output

def _compute_trace(op_cls, *args, **kwargs):
    args = tuple(
        _to_Tensor(arg, getattr(arg, "requires_grad", False)) if isinstance(arg, Fensor) else arg
        for arg in args
    )

    kwargs = {k: (_to_Tensor(v, getattr(v, "requires_grad", False)) if isinstance(v, Fensor) else v) for k, v in kwargs.items()}

    output = Tensor._from_op(op_cls, )
