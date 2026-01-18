from deepen.backend import active_backend as bx
from deepen.ops.ewise_ops import *
from deepen.ops.logical_ops import *
from deepen.ops.index_ops import *
from deepen.ops.shape_ops import *
from deepen.ops.reduction_ops import *
from deepen.ops.linalg_ops import *
from deepen.ops.activation_ops import *
from deepen.ops.utils import _make_cache, _compute_initializer_fans

_bx = bx() # backend singleton

class Tensor:
    _eager_mode = False
    _no_grad_mode = False
    
    def __init__(self, data=None, requires_grad=True):
        if data is not None:
            data = _bx.asarray(data)

        if Tensor._eager_mode or Tensor._no_grad_mode:
            requires_grad = False

        # primary tensor attributes
        self.data = data
        self.grad = None
        self.requires_grad = requires_grad

        # attributes for the computation graph
        self._op = None # the operation that made this tensor
        self._parents = () # a tuple of parent tensors
        self._save = None # intermediate values stored in an operation class' forward (needed by its backward)
        self._args = None # arguments that are directly involved in an operation (things like actual tensor data, etc)
        self._kwargs = None # arguments that aren't like the above (things like axes, shape, etc)

        # attributes for function decorators
        self._is_immutable = False
    
    def __hash__(self): # __eq__ is used as a logical operation so we need to define __hash__
        return id(self)

    @property
    def size(self):
        if self.data is None:
            raise AttributeError("cannot access .size for a Tensor with no data")
        return self.data.size

    @property
    def shape(self):
        if self.data is None:
            raise AttributeError("cannot access .shape for a Tensor with no data")
        return self.data.shape
    
    @property   
    def ndim(self):
        if self.data is None:
            raise AttributeError("cannot access .ndim for a Tensor with no data")
        return self.data.ndim

    @property
    def dtype(self):
        if self.data is None:
            raise AttributeError("cannot access .dtype for a Tensor with no data")
        return self.data.dtype

    def __getitem__(self, key):
        if self.data is None:
            raise AttributeError("cannot access .__getitem__ for a Tensor with no data")
        view = object.__new__(Tensor) # create a simple view
        view.data = self.data[key]
        view.grad = self.grad[key] if self.grad is not None else None
        view.requires_grad = False
        view._is_immutable = self._is_immutable
        return view

    def __setitem__(self, key, value):
        if self._is_immutable:
            raise ValueError("cannot modify an immutable Tensor")
        if self.data is None:
            raise AttributeError("cannot access .__setitem__ for a Tensor with no data")
        self.data[key] = value
    
    def __repr__(self):
        return str(self.data)

    def __array__(self, dtype=None):
        return _bx.asarray(self.data, dtype=dtype)
    
    @staticmethod
    def _from_op(op_cls, *args, **kwargs):
        args_list = []
        
        for arg in args:
            if isinstance(arg, Tensor):
                args_list.append((True, arg)) # placeholder for a Tensor (True)
            elif isinstance(arg, (tuple, list)) and all(isinstance(e, (int, float, bool)) for e in arg):
                tup = tuple(arg)
                args_list.append((False, tup)) # placeholder for static metadata (False)
            elif isinstance(arg, (int, float, bool)) or isinstance(arg, _bx.ndarray):
                arr = arg if isinstance(arg, _bx.ndarray) else _bx.array(arg)
                args_list.append((False, arr)) # placeholder for numeric literals (False)
            else:
                raise ValueError(f"unexpected positional argument {arg!r} of type {type(arg)}")
        
        kwargs = {k: tuple(v) if isinstance(v, list) else v for k, v in kwargs.items()} # list mutability can be unsafe

        parents = tuple(arg for is_tensor, arg in args_list if is_tensor) # extract parent Tensors
        requires_grad = any(parent.requires_grad for parent in parents)

        if Tensor._eager_mode:
            args = [arg.data if isinstance(arg, Tensor) else arg for arg in args]

            output = Tensor(data=None, requires_grad=False)
            output._save = _make_cache(op_cls, False)
            output.data = op_cls.forward(output._save, *args, **kwargs)

            return output

        output = Tensor(data=None, requires_grad=requires_grad)
        output._op = op_cls
        output._parents = parents
        output._save = _make_cache(op_cls, True)
        output._args = tuple(args_list)
        output._kwargs = kwargs

        return output

    # Hidden helper functions
    def _reset_grad(self):
        self.grad = None

    def _has_no_op(self):
        return self._op is None

    def _can_send_grad(self): # checks if a tensor can send gradients to its parents
        return self.requires_grad and self.grad is not None and self._op is not None

    def _can_receive_grad(self, grad): # checks if a tensor can recieve gradients from its children
        return self.requires_grad and grad is not None

    # Exposed helper functions
    def detach(self):
        return Tensor(self.data, requires_grad=False)
    
    # Tensor creation/initialization methods
    @classmethod
    def zeros(cls, shape, dtype=_bx.float32, requires_grad=True):
        return cls(_bx.zeros(shape=shape, dtype=dtype), requires_grad=requires_grad)

    @classmethod
    def ones(cls, shape, dtype=_bx.float32, requires_grad=True):
        return cls(_bx.ones(shape=shape, dtype=dtype), requires_grad=requires_grad)

    @classmethod
    def constant(cls, shape, value, dtype=_bx.float32, requires_grad=True):
        return cls(_bx.full(shape=shape, fill_value=value, dtype=dtype), requires_grad=requires_grad)
    
    @classmethod
    def random(cls, shape, dtype=_bx.float32, requires_grad=True):
        return cls(_bx.random.random(size=shape, dtype=dtype), requires_grad=requires_grad)
    
    @classmethod
    def uniform(cls, shape, bounds=(0.0, 1.0), dtype=_bx.float32, requires_grad=True):
        return cls(_bx.random.uniform(size=shape, low=bounds[0], high=bounds[1], dtype=dtype), requires_grad=requires_grad)
    
    @classmethod
    def normal(cls, shape, mean=0.0, std=1.0, dtype=_bx.float32, requires_grad=True):
        return cls(_bx.random.normal(size=shape, loc=mean, scale=std, dtype=dtype), requires_grad=requires_grad)
    
    @classmethod
    def xavier_uniform(cls, shape, dtype=_bx.float32, requires_grad=True):
        fan_in, fan_out = _compute_initializer_fans(shape)
        limit = _bx.sqrt(6 / (fan_in + fan_out))
        return cls(_bx.random.uniform(size=shape, low=-limit, high=limit, dtype=dtype), requires_grad=requires_grad)

    @classmethod
    def xavier_normal(cls, shape, dtype=_bx.float32, requires_grad=True):
        fan_in, fan_out = _compute_initializer_fans(shape)
        scale = _bx.sqrt(2 / (fan_in + fan_out))
        return cls(_bx.random.normal(size=shape, loc=0.0, scale=scale, dtype=dtype), requires_grad=requires_grad)

    @classmethod
    def he_uniform(cls, shape, dtype=_bx.float32, requires_grad=True):
        fan_in, _ = _compute_initializer_fans(shape)
        limit = _bx.sqrt(6 / (fan_in))
        return cls(_bx.random.uniform(size=shape, low=-limit, high=limit, dtype=dtype), requires_grad=requires_grad)

    @classmethod
    def he_normal(cls, shape, dtype=_bx.float32, requires_grad=True):
        fan_in, _ = _compute_initializer_fans(shape)
        scale = _bx.sqrt(2 / (fan_in))
        return cls(_bx.random.normal(size=shape, loc=0.0, scale=scale, dtype=dtype), requires_grad=requires_grad)

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

    # Index operations
    def gather(self, indices=None): return Tensor._from_op(gather, self, indices=indices)

    # Shape operations
    def squeeze(self, axes=None): return Tensor._from_op(squeeze, self, axes=axes)
    def unsqueeze(self, axes=None): return Tensor._from_op(unsqueeze, self, axes=axes)
    def transpose(self, axes=None): return Tensor._from_op(transpose, self, axes=axes)
    def concatenate(self, other, axes=None): return Tensor._from_op(concatenate, self, other, axes=axes)
    def reshape(self, *shape): return Tensor._from_op(reshape, self, shape=shape)

    # Reduction operations
    def sum(self, axes=None): return Tensor._from_op(sum_, self, axes=axes)
    def mean(self, axes=None): return Tensor._from_op(mean, self, axes=axes)
    def min(self, axes=None): return Tensor._from_op(min_, self, axes=axes)
    def max(self, axes=None): return Tensor._from_op(max_, self, axes=axes)
    def softmax(self, axes=None): return Tensor._from_op(softmax, self, axes=axes)

    # Linear algebra operations
    def matmul(self, other): return Tensor._from_op(matmul, self, other)
    def outer(self, other): return Tensor._from_op(outer, self, other)

    # Activation functions
    def sigmoid(self): return Tensor._from_op(sigmoid, self)
    def tanh(self): return Tensor._from_op(tanh, self)
    def relu(self): return Tensor._from_op(relu, self)
    def leaky_relu(self, neg_slope=0.1): return Tensor._from_op(leaky_relu, self, neg_slope=neg_slope)
    def swish(self): return Tensor._from_op(swish, self)

class Parameter(Tensor):
    def __init__(self, data=None, requires_grad=True):
        if Tensor._eager_mode:
            raise ValueError("cannot create/modify Parameter(s) in eager mode")
        if Tensor._no_grad_mode:
            raise ValueError("cannot create/modify Parameter(s) in no_grad mode")

        super().__init__(data, requires_grad)

        self._is_parameter = True