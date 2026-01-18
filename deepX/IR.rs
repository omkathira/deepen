use serde::{Serialize, Deserialize};
use std::collections::HashMap;

type TensorId = usize;
type NodeId = usize;

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
enum DataType {
    Int32,
    Int64,
    Float32,
    Float64,
    Bool
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
enum TensorType {
    Input, // tensors from feed_dict
    Output, // output tensor
    Intermediate, // tensors from intermediate computations
    Parameter // learnable parameter tensors
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
enum OpAttr {
    Int(i64),
    Float(f64),
    IntList(Vec<i64>),
    FloatList(Vec<f64>),
    String(String),
    Bool(bool)
}

// Tensor IR
#[derive(Serialize, Deserialize, Debug, Clone)]
struct Tensor {
    id: TensorId,
    shape: Vec<usize>,
    requires_grad: bool,
    tensor_type: TensorType
}

// Node IR
#[derive(Serialize, Deserialize, Debug, Clone)]
struct Node {
    id: NodeId,
    inputs: Vec<TensorId>,
    output: TensorId,
    op_name: String,
    op_attrs: HashMap<String, OpAttr> // kwargs
}

// Graph IR
#[derive(Serialize, Deserialize, Debug, Clone)]
struct Graph {
    version: String,
    tensors: HashMap<TensorId, Tensor>,
    nodes: Vec<Node>,
    inputs: Vec<TensorId>,
    output: TensorId,
    topo_order: Vec<NodeId> // execution order
}

// Compilation Configuration
#[derive(Serialize, Deserialize, Debug, Clone)]
struct CompilationConfig {
    dtype: DataType,
    mixed_precision: bool
}