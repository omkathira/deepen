from deepen.core.tensor import Tensor

# Eager mode
class _EagerMode:
    def __init__(self):
        self._prev_state = False
    
    def __enter__(self):
        self._prev_state = Tensor._eager_mode
        Tensor._eager_mode = True
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        Tensor._eager_mode = self._prev_state

def eager():
    return _EagerMode()

# No grad mode
class _NoGradMode:
    def __init__(self):
        self._prev_state = False
    
    def __enter__(self):
        self._prev_state = Tensor._no_grad_mode
        Tensor._no_grad_mode = True
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        Tensor._no_grad_mode = self._prev_state

def no_grad():
    return _NoGradMode()

__all__ = ['eager', 'no_grad']