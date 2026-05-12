#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <variant>
#include <unordered_map>
#include <nlohmann/json.hpp>

namespace deepx {

using TensorId = std::size_t;
using NodeId   = std::size_t;

enum class DataType { Int32, Int64, Float32, Float64, Bool };
enum class TensorType { Input, Output, Intermediate, Parameter };

using OpAttr = std::variant<
    std::int64_t,
    double,
    std::vector<std::int64_t>,
    std::vector<double>,
    std::string,
    bool>;

struct Tensor {
    TensorId id;
    std::vector<std::size_t> shape;
    bool requires_grad;
    TensorType tensor_type;
};

struct Node {
    NodeId id;
    std::vector<TensorId> inputs;
    TensorId output;
    std::string op_name;
    std::unordered_map<std::string, OpAttr> op_attrs;
};

struct Graph {
    std::string version;
    std::unordered_map<TensorId, Tensor> tensors;
    std::vector<Node> nodes;
    std::vector<TensorId> inputs;
    TensorId output;
    std::vector<NodeId> topo_order;

    static Graph from_json(const nlohmann::json& j);
    nlohmann::json to_json() const;
};

struct CompilationConfig {
    DataType dtype;
    bool mixed_precision;
};

} // namespace deepx
