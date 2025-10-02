from deepen.backend import active_backend as bx
from deepen.core.tensor import Tensor

_bx = bx() # backend singleton

class Graph:
    def __init__(self, root: Tensor):
        self._clear_computed_data(root)
        self._nodes = self._topo_sort(root)

    def _clear_computed_data(self, t):
        if t._op is not None:
            t.data = None
            for parent in t._parents:
                self._clear_computed_data(parent)

    def _zero_grad(self):
        for t in self._nodes:
            t._reset_grad()

    def _traverse(self, t: Tensor, visited, topo_order):
        if t in visited or t._has_no_parents(): # ignore tensors that weren't generated from an upstream operation
            return
        
        visited.add(t)
        
        for parent in t._parents:
            self._traverse(parent, visited, topo_order)

        topo_order.append(t)

    def _topo_sort(self, root: Tensor): # build a node dependency-based execution order
        visited = set()
        topo_order = []

        self._traverse(root, visited, topo_order)

        return topo_order

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
                if not parent._can_receive_grad(grad): # tensor doesn't track gradients
                    continue
                if parent.grad is None: # first time accumulating gradients
                    parent.grad = grad
                else:
                    parent.grad += grad # updating gradients

    def run(self, feed_dict):
        for ph_t, data in feed_dict.items():
            ph_t.data = data

        self._zero_grad()

        self._forward()

        self._nodes[-1].grad = _bx.ones_like(self._nodes[-1].data) # seed loss

        self._backward()

        return self._nodes[-1]