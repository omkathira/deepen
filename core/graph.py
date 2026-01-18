import json
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
    
    def _serialize_op_attr(self, value):
        if isinstance(value, int): return {"Int": value}
        elif isinstance(value, float): return {"Float": value}
        elif isinstance(value, (list, tuple)):
            if all(isinstance(x, int) for x in value): return {"IntList": list(value)}
            elif all(isinstance(x, float) for x in value): return {"FloatList": list(value)}
            raise ValueError(f"mixed types in: {value}") # in case someone does something like [1, "4"]
        elif isinstance(value, bool): return {"Bool": value}
        elif isinstance(value, str): return {"String": value}
        elif value is None: return {"String": "None"}
        else: raise ValueError(f"unsupported op attribute type: {type(value)}")

    def _serialize(self, root: Tensor, sample_inputs: dict, dtype="Float32"):
        for t, data in sample_inputs.items(): # feed sample inputs (for shape information)
            t.data = data
        
        self._forward() # populate shapes for all tensors

        tensors_IR = {}
        inputs = []
        
        for t_id, node in self._nodes.items():
            t = node.tensor

            if t._op is None: # figure out what type of tensor we're working with
                if hasattr(t, '_is_parameter') and t._is_parameter:
                    tensor_type = "Parameter"
                else:
                    tensor_type = "Input"
                    inputs.append(t_id)
            else:
                tensor_type = "Intermediate"
            
            if t.data is not None:
                shape = list(t.shape)
            else:
                shape = []

            tensors_IR[t_id] = {
                "id": t_id,
                "shape": shape,
                "requires_grad": t.requires_grad,
                "tensor_type": tensor_type,
            }

        output_id = id(root)
        if output_id in tensors_IR: # mark the output tensor (the loss)
            tensors_IR[output_id]["tensor_type"] = "Output"
        
        nodes_IR = []
        node_id = 0
        topo_order = []

        for t_id in self._topo_order:
            t = self._nodes[t_id].tensor

            if t._has_no_op():
                continue

            op_name = t._op.__name__
            op_attrs = {}
            for k, v in (t._kwargs or {}).items():
                op_attrs[k] = self._serialize_op_attr(v)
            
            node_IR = {
                "id": node_id,
                "inputs": [id(p) for p in t._parents],
                "output": t_id,
                "op_name": op_name,
                "op_attrs": op_attrs,
            }

            nodes_IR.append(node_IR)
            topo_order.append(node_id)
            node_id += 1
        
        graph = {
            "version": "1.0",
            "tensors": tensors_IR,
            "nodes": nodes_IR,
            "inputs": inputs,
            "output": output_id,
            "topo_order": topo_order,
        }

        return json.dumps(graph, indent=2)
    
    def compile(self):
        pass

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