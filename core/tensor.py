from numbers import Number
from deepen.backend import active_backend as bx
from deepen.ops.ewise_ops import *
from deepen.ops.logical_ops import *
from deepen.ops.reduce_ops import *
from deepen.ops.shape_ops import *
from deepen.ops.linalg_ops import *
from deepen.ops.activation_ops import *
from deepen.ops.utils import Cache

_bx = bx() # backend singleton

class Tensor:
    def __init__(self, data, requires_grad=False):
        if data is not None and not isinstance(data, _bx.ndarray):
            data = _bx.array(data)

        # tensor attributes (accessible to the user)
        self.data = data
        self.grad = None
        self.requires_grad = requires_grad

        # important attributes for the computation graph
        self._op = None # the operation that made this tensor
        self._parents = None # a tuple of parent tensors
        self._save = None # intermediate values stored in an operation class' forward (needed by its backward)
        self._args = None # arguments that are data we pass through the neural network
        self._kwargs = None # arguments that aren't like the above (like axes, shape, etc)
    
    def __hash__(self): # __eq__ is a logical operation so we need to define __hash__
        return id(self)

    @property
    def shape(self):
        if self.data is None:
            raise AttributeError("cannot access .shape for a Tensor with no data")
        return self.data.shape
    
    @staticmethod
    def _from_op(op_cls, *args, **kwargs):
        kwargs = {
            k: tuple(v) if isinstance(v, list) else v # list mutability can be unsafe
            for k, v in kwargs.items()
        }

        args_list = []
        
        for arg in args:
            if isinstance(arg, Tensor): # placeholder for a Tensor (True)
                args_list.append((True, arg))
            elif isinstance(arg, Number) or isinstance(arg, _bx.ndarray): # placeholder for a literal (False)
                arr = arg if isinstance(arg, _bx.ndarray) else _bx.array(arg)
                args_list.append((False, arr))
            else:
                raise ValueError(f"unexpected positional argument {arg!r} of type {type(arg)}")

        parents = tuple(t for is_ph, t in args_list if is_ph) # extract parent Tensors
        requires_grad = any(p.requires_grad for p in parents)

        output = Tensor(data=None, requires_grad=requires_grad)
        output._op = op_cls
        output._parents = parents
        output._save = Cache()
        output._args = tuple(args_list)
        output._kwargs = kwargs

        return output

    # Internal helper functions
    def _reset_grad(self):
        self.grad = None

    def _has_no_parents(self):
        return True if self._op is None else False

    def _can_backprop(self):
        return self.requires_grad and self.grad is not None

    def _can_receive_grad(self, grad):
        return self.requires_grad and grad is not None

    # Exposed helper functions
    def detach(self):
        return Tensor(self.data, requires_grad=False)
    
    # Tensor creation/initialization methods
    @staticmethod
    def zeros(shape, dtype=_bx.float32, requires_grad=False):
        return Tensor(_bx.zeros(shape=shape, dtype=dtype), requires_grad=requires_grad)

    @staticmethod
    def ones(shape, dtype=_bx.float32, requires_grad=False):
        return Tensor(_bx.ones(shape=shape, dtype=dtype), requires_grad=requires_grad)

    @staticmethod
    def constant(shape, value, dtype=_bx.float32, requires_grad=False):
        return Tensor(_bx.full(shape=shape, fill_value=value, dtype=dtype), requires_grad=requires_grad)
    
    @staticmethod
    def random(shape, dtype=_bx.float32, requires_grad=False):
        return Tensor(_bx.random.random(size=shape, dtype=dtype), requires_grad=requires_grad)
    
    @staticmethod
    def uniform(shape, interval=(0.0, 1.0), dtype=_bx.float32, requires_grad=False):
        return Tensor(_bx.random.uniform(size=shape, low=interval[0], high=interval[1], dtype=dtype), requires_grad=requires_grad)
    
    @staticmethod
    def normal(shape, loc=0.0, scale=1.0, dtype=_bx.float32, requires_grad=False):
        return Tensor(_bx.random.normal(size=shape, loc=loc, scale=scale, dtype=dtype), requires_grad=requires_grad)
    
    @staticmethod
    def xavier(shape, dist_type="uniform", dtype=_bx.float32, requires_grad=False):
        fan_in, fan_out = shape

        if dist_type == "uniform":
            limit = _bx.sqrt(6 / (fan_in + fan_out))
            return Tensor(_bx.random.uniform(size=shape, low=-limit, high=limit, dtype=dtype), requires_grad=requires_grad)
        
        elif dist_type == "normal":
            scale = _bx.sqrt(2 / (fan_in + fan_out))
            return Tensor(_bx.random.normal(size=shape, loc=0.0, scale=scale, dtype=dtype), requires_grad=requires_grad)

    @staticmethod
    def he(shape, dist_type="uniform", dtype=_bx.float32, requires_grad=False):
        fan_in, _ = shape

        if dist_type == "uniform":
            limit = _bx.sqrt(6 / (fan_in))
            return Tensor(_bx.random.uniform(size=shape, low=-limit, high=limit, dtype=dtype), requires_grad=requires_grad)
        
        elif dist_type == "normal":
            scale = _bx.sqrt(2 / (fan_in))
            return Tensor(_bx.random.normal(size=shape, loc=0.0, scale=scale, dtype=dtype), requires_grad=requires_grad)
    
    # Element-wise operations
    def __add__(self, other): return Tensor._from_op(add, self, other)
    def __radd__(self, other): return Tensor._from_op(add, other, self)
    def __sub__(self, other): return Tensor._from_op(sub, self, other)
    def __rsub__(self, other): return Tensor._from_op(sub, other, self)
    def __mul__(self, other): return Tensor._from_op(mul, self, other)
    def __rmul__(self, other): return Tensor._from_op(mul, other, self)
    def __truediv__(self, other): return Tensor._from_op(div, self, other)
    def __rtruediv__(self, other): return Tensor._from_op(div, other, self)
    def __neg__(self): return Tensor._from_op(neg, self)
    def __abs__(self): return Tensor._from_op(abs_, self)
    def __pow__(self, power): return Tensor._from_op(pow_, self, power)
    def exp(self): return Tensor._from_op(exp, self)
    def log(self, base=None): return Tensor._from_op(log, self, base=base)
    def clip(self, min_val, max_val): return Tensor._from_op(clip, self, min_val=min_val, max_val=max_val)

    # Logical operations
    def __eq__(self, other): return Tensor._from_op(eq, self, other)
    def __ne__(self, other): return Tensor._from_op(ne, self, other)
    def __lt__(self, other): return Tensor._from_op(lt, self, other)
    def __le__(self, other): return Tensor._from_op(le, self, other)
    def __gt__(self, other): return Tensor._from_op(gt, self, other)
    def __ge__(self, other): return Tensor._from_op(ge, self, other)
    def __invert__(self): return Tensor._from_op(not_, self)
    def __and__(self, other): return Tensor._from_op(and_, self, other)
    def __or__(self, other): return Tensor._from_op(or_, self, other)

    # Reduction operations
    def sum(self, axes=None): return Tensor._from_op(sum_, self, axes=axes)
    def mean(self, axes=None): return Tensor._from_op(mean, self, axes=axes)
    def min(self, axes=None): return Tensor._from_op(min_, self, axes=axes)
    def max(self, axes=None): return Tensor._from_op(max_, self, axes=axes)
    def softmax(self, axes=None): return Tensor._from_op(softmax, self, axes=axes)

    # Shape operations
    def squeeze(self, axes=None): return Tensor._from_op(squeeze, self, axes=axes)
    def unsqueeze(self, axes=None): return Tensor._from_op(unsqueeze, self, axes=axes)
    def transpose(self, axes=None): return Tensor._from_op(transpose, self, axes=axes)
    def concatenate(self, other, axes=None): return Tensor._from_op(concatenate, self, other, axes=axes)
    def reshape(self, *shape): return Tensor._from_op(reshape, self, shape=shape)

    # Linear algebra operations
    def matmul(self, other): return Tensor._from_op(matmul, self, other)
    def outer(self, other): return Tensor._from_op(outer, self, other)

    # Activation functions
    def sigmoid(self): return Tensor._from_op(sigmoid, self)
    def tanh(self): return Tensor._from_op(tanh, self)
    def relu(self): return Tensor._from_op(relu, self)
    def leaky_relu(self, neg_slope=0.1): return Tensor._from_op(leaky_relu, self, neg_slope=neg_slope)