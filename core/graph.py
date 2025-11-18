from deepen.backend import active_backend as bx
from deepen.core.tensor import Tensor

_bx = bx() # backend singleton

class _Node:
    def __init__(self, t: Tensor):
        self.tensor = t
        self.parents = tuple(id(parent) for parent in t._parents)
        self.children = []

class Graph:
    def __init__(self, root: Tensor):
        self._nodes = {}
        self._topo_order = []

        self._clear_computed_data(root)
        self._topo_sort(root)

    def _clear_computed_data(self, t):
        if t._op is not None:
            t.data = None
            for parent in t._parents:
                self._clear_computed_data(parent)

    def _traverse(self, t: Tensor, visited):
        if id(t) in visited:
            return
        
        visited.add(id(t))
        
        for parent in t._parents:
            self._traverse(parent, visited)

        self._nodes[id(t)] = _Node(t)

        self._topo_order.append(id(t))

        for parent in t._parents:
            self._nodes[id(parent)].children.append(id(t))

    def _topo_sort(self, root: Tensor): # build the execution order
        visited = set()
        self._traverse(root, visited)

    def _zero_grad(self):
        for t_id in self._topo_order:
            self._nodes[t_id].tensor._reset_grad()

    def _forward(self):
        for t_id in self._topo_order:
            t = self._nodes[t_id].tensor

            if t._has_no_op():
                continue
            
            args = [arg.data if is_tensor else arg for is_tensor, arg in t._args] # rebuild positional arguments

            t.data = t._op.forward(t._save, *args, **t._kwargs)

    def _backward(self):
        for t_id in reversed(self._topo_order):
            t = self._nodes[t_id].tensor

            if not t._can_send_grad():
                continue

            grads = t._op.backward(t._save, t.grad)

            for parent, grad in zip(t._parents, grads):
                if not parent._can_receive_grad(grad): # tensor doesn't track gradients
                    continue
                if parent.grad is None: # first time accumulating gradients
                    parent.grad = grad
                else:
                    parent.grad += grad # accumulating gradients

    def run(self, feed_dict):
        for ph_t, data in feed_dict.items():
            ph_t.data = data

        self._zero_grad()

        self._forward()

        root_t_id = self._topo_order[-1]
        root_t = self._nodes[root_t_id].tensor

        if Tensor._eager_mode:
            return root_t

        if root_t.requires_grad:
            root_t.grad = _bx.ones_like(root_t.data) # seed loss

            self._backward()

        return root_t