from deepen.backend import active_backend as bx
from deepen.core.tensor import Tensor

_bx = bx() # backend singleton

class Graph:
    def __init__(self, root: Tensor):
        self._nodes = self._topo_sort(root)

    def _topo_sort(self, root: Tensor): # build a node dependency-based execution order
        visited = set()
        topo_order = []

        def traverse(t: Tensor):
            if t in visited or t._has_no_parents(): # ignore tensors that weren't generated from an upstream operation
                return
            
            visited.add(t)
            
            for p in t._parents:
                traverse(p)

            topo_order.append(t)

        traverse(root)

        return topo_order
    
    def _zero_grad(self):
        for t in self._nodes:
            t._reset_grad()

    def _forward(self):
        for t in self._nodes:
            if t._has_no_parents():
                continue
            
            # rebuild positional arguments
            args = [
                arg.data if is_tensor else arg
                for is_tensor, arg in t._args
            ]

            t.data = t._op.forward(t._save, *args, **t._kwargs) # execute the operation's forward method

    def _backward(self):
        for t in reversed(self._nodes):
            if not t._can_backprop():
                continue

            grads = t._op.backward(t._save, t.grad)

            for parent, grad in zip(t._parents, grads):
                if not parent._can_receive_grad(grad): # tensor doesn't track a gradient
                    continue
                if parent.grad is None: # first time accumulating a gradient
                    parent.grad = grad
                else:
                    parent.grad = parent.grad + grad # updating gradient

    def run(self, feed_dict):
        for ph_t, data in feed_dict.items():
            ph_t.data = data

        self._zero_grad()

        self._forward()

        self._nodes[-1].grad = _bx.ones_like(self._nodes[-1].data) # seed loss gradient

        self._backward()

        return self._nodes[-1]