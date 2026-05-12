from deepen.core.tensor import Tensor
from deepen.core.graph import Graph

def _make_immutable(args, kwargs):
    immutable_args = []

    for arg in args:
        if isinstance(arg, Tensor):
            t_copy = Tensor(arg.data, requires_grad=True)
            t_copy._is_immutable = True
            immutable_args.append(t_copy)
        else:
            immutable_args.append(arg)
    
    immutable_kwargs = {}
    
    for key, value in kwargs.items():
        if isinstance(value, Tensor):
            t_copy = Tensor(value.data, requires_grad=True)
            t_copy._is_immutable = True
            immutable_kwargs[key] = t_copy
        else:
            immutable_kwargs[key] = value
    
    return immutable_args, immutable_kwargs

def trace(f):
    def wrapper(*args, **kwargs):
        immutable_args, immutable_kwargs = _make_immutable(args, kwargs)
        
        result = f(*immutable_args, **immutable_kwargs)
        result.requires_grad = True
        
        graph = Graph(result)
        feed_dict = {arg: arg.data for arg in immutable_args if isinstance(arg, Tensor)}
        graph.run(feed_dict)

        return result
    return wrapper

def grad(f):
    def wrapper(*args, **kwargs):
        immutable_args, immutable_kwargs = _make_immutable(args, kwargs)
        
        result = f(*immutable_args, **immutable_kwargs)
        result.requires_grad = True
        
        graph = Graph(result)
        feed_dict = {arg: arg.data for arg in immutable_args if isinstance(arg, Tensor)}
        graph.run(feed_dict)
        
        if len(immutable_args) == 1:
            return immutable_args[0].grad
        else:
            return tuple(arg.grad for arg in immutable_args if isinstance(arg, Tensor))
            
    return wrapper