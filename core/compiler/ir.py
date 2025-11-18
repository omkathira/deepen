from deepen.core.tensor import Tensor, Parameter
from deepen.core.graph import Graph

class TensorIR:
    def __init__(self, tensor_id, shape, dtype, constant, data_ref, name):
        self.tensor_id = tensor_id
        self.shape = shape
        self.dtype = dtype
        self.constant = constant
        self.data_ref = data_ref
        self.name = name

class NodeIR:
    def __init__(self, node_id, op, inputs, outputs, attributes):
        self.node_id = node_id
        self.op = op
        self.inputs = inputs
        self.outputs = outputs
        self.attributes = attributes

    def to_json(self):
        attributes = {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.attributes.items()}
        return {
            "id": self.node_id,
            "op": self.op,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "attributes": attributes
        }

class GraphIR:
    def __init__(self, version, inputs, outputs, tensors, nodes):
        self.version = version
        self.inputs = inputs
        self.outputs = outputs
        self.tensors = tensors
        self.nodes = nodes

    def to_json(self):
        return {
            "graph": {
                "version": self.version,
                "inputs": self.inputs,
                "outputs": self.outputs,
                "tensors": [{"id": t.id, "shape": t.shape, "dtype": t.dtype, "constant": t.constant, "data_ref": t.data_ref, "name": t.name} for t in self.tensors],
                "nodes": [node.to_json() for node in self.nodes],
            }
        }

