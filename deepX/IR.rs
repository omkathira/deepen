type TensorId = usize;
type NodeId = usize;

// Tensor IR
struct Tensor {
    id: TensorId,
    shape: Vec<usize>,
    dtype: DType,
    requires_grad: bool,
    tensor_type: TensorType
}

// Node IR
struct Node {
    id: NodeId,
    op: Op,
    inputs: Vec<TensorId>,
    outputs: TensorId,
    op_attrs: HashMap<>
}

// Graph IR
struct Graph {
    version: String,
    tensors: HashMap<TensorId, Tensor>,
    nodes: Vec<Node>,
    inputs: Vec<TensorId>,
    outputs: TensorId,
    
}