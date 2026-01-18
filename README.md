<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/othneildrew/Best-README-Template">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>

  <h1 align="center">Deepen</h1>

  <p align="center">
    A compilation-enabled, static-graph deep learning framework.
  </p>
</div>

<!-- ABOUT THE PROJECT -->
## About

Deepen is a custom, static-graph deep learning framework that implements automatic differentiation, computational graphs, and a complete suite of neural network primitives. Inspired by PyTorch and JAX, it provides a flexible, abstracted tensor interface with dual execution modes - lazy graph building by default and eager execution for debugging - along with a full reverse-mode autodiff system for training neural networks via backpropagation. The framework features a pluggable backend system supporting both NumPy (CPU) and CuPy (GPU), comprehensive neural network layers (Linear, Conv2d, BatchNorm, etc), popular optimizers (SGD, RMSprop, Adam, etc), and common loss functions.

I'm currently working on a Rust-based compiler (DeepX) with graph serialization to an SSA-based IR for model compilation. The compiler is primarily focused on implementing graph-level optimization passes (DCE, CSE, etc) and operator fusion. This system will route to a custom CUDA backend built using cuBLAS/cuDNN and fused CUDA kernels.

## Framework Structure

**Core Infrastructure** (core/)
```
tensor.py --> tensor class with autograd, gradient tracking, operator overloading, weight initializers (Xavier, He)
graph.py --> computation graph builder/executor with topological sorting, forward/backward passes, graph serialization
decorators.py --> @trace (captures computation graph from a function), @grad (returns a gradient function)
context.py --> context managers - eager() mode (for debugging), no_grad() mode (for inference)
```
**Op modules** (ops/)
```
ewise_ops.py --> add, sub, mul, div, neg, abs, pow, exp, log, clip
logical_ops.py --> eq, ne, lt, le, gt, ge, not_, and_, or_
shape_ops.py --> squeeze, unsqueeze, transpose, concatenate, reshape
reduction_ops.py --> sum, mean, min, max, softmax
linalg_ops.py --> matmul, outer, _im2col, _col2im
activation_ops.py --> sigmoid, tanh, relu, leaky_relu, swish
stochastic_ops.py --> dropout, gaussian_noise
index_ops.py --> gather (advanced indexing for Transformers)
utils.py --> gradient broadcasting, axes normalization, initializer helpers
```
**High-Level API**
```
layers.py --> neural network layers - Linear, Conv2d, MaxPool2d, BatchNorm1d/2d, Dropout, etc
compose.py --> sequential container, activation wrappers, model blocks (residuals, and support for RNNs, CNNs, Transformers, etc)
losses.py --> MSE, MAE, binary cross-entropy, cross-entropy, KL divergence (planned)
optimizers.py --> SGD (with momentum), RMSprop, Adam, AdamW, Muon (planned)
backend.py --> backend abstraction for NumPy/CuPy switching
```
**Compiler Infrastructure** (deepX/)
```
IR.rs --> rust IR definitions - Tensor, Node, Graph structs, etc
compiler.rs --> compiler implementation (in progress)
cuda/ --> cuBLAS/cuDNN and fused CUDA kernel backend (in progress)
```
<!-- GETTING STARTED -->
## Getting Started

I highly recommend using micromamba to setup Deepen's environment. Installing micromamba is super simple ([instructions here](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html)). Currently, Deepen's GPU backend only works on Linux/Windows (you must be able to use ```CuPy```). The compiler backend is GPU-only (uses NVIDIA's CUDA libraries and CUDA C/C++ code directly).

```
micromamba create -n deepen -c conda-forge python=3.13 numpy cupy ipykernel
```

You can pick between installing either ```NumPy``` or ```CuPy```. The last package, ```ipykernel``` is generally useful as it lets you run code in Jupyter-style notebooks in VSCode/Cursor. Eventually, I'll update this to include instructions on how to setup ```rust```, ```cuda```, and their related packages once the compiler is ready.

## Example

```python
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt

import deepen as dpn
from deepen.core.tensor import Tensor # activations are intrinsic to tensors!
from deepen.core.graph import Graph
from deepen.layers import Linear, BatchNorm1d

# define some mock data that follows a sine curve
N = 100

X_data = cp.linspace(0, 1, N, dtype=cp.float32).reshape(-1, 1)
Y_data = cp.sin(2 * cp.pi * X_data) + 0.1 * cp.random.randn(N, 1).astype(cp.float32)

X = dpn.Tensor(data=None, requires_grad=False)
Y = dpn.Tensor(data=None, requires_grad=False)

feed_dict = {X: X_data, Y: Y_data}

# an overly complex neural network to model sine b/c why not
class SineNet(dpn.Layer):
    def __init__(self):
        super().__init__()
        self.l1 = dpn.Linear(1, 64, weight_init="he_uniform")
        self.norm1 = dpn.BatchNorm1d(64)
        self.l2 = dpn.Linear(64, 64, weight_init="he_uniform")
        self.norm2 = dpn.BatchNorm1d(64)
        self.l3 = dpn.Linear(64, 1, weight_init="he_uniform")
    
    def forward(self, x):
        x = self.l1(x).swish()
        x = self.norm1(x)
        x = self.l2(x).swish()
        x = self.norm2(x)
        return self.l3(x)
    
    def build(self, X, Y):
        pred = self.forward(X)
        loss = dpn.mse(pred, Y)
        comp_graph = dpn.Graph(loss)
        return comp_graph, pred, loss
        
snet = SineNet()
model, pred, loss = snet.build(X, Y)
optimizer = dpn.Adam(snet.parameters(), lr=0.01)

losses = []
for epoch in range(1, 301):
    optimizer.zero_grad()
    loss = model.run(feed_dict)
    optimizer.step()
    losses.append(float(loss.data))
    if epoch % 50 == 0: # print out the model's loss every 50 epochs
        print(f"epoch {epoch:4d}, loss {losses[-1]:.4f}")

# bring stuff back to NumPy b/c matplotlib can't work with CuPy arrays
X_np = X_data.get()
Y_np = Y_data.get()
pred_np = pred.data.get()

# quick and dirty plot
plt.figure(figsize=(4,3))
plt.scatter(X_np, Y_np, s=8, alpha=0.4, label="data")
plt.plot(X_np, pred_np, color="r", lw=2, label="model")
plt.xlabel("x")
plt.ylabel("y = sin(x)")
plt.legend()
plt.title("Fitting a Simple Sine!")
plt.tight_layout()
plt.show()
```

<!-- LICENSE -->
## License

MIT License

Copyright (c) 2026 Om Kathira

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
