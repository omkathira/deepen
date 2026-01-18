<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/othneildrew/Best-README-Template">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">Deepen</h3>

  <p align="center">
    A compilation-enabled, static-graph deep learning framework.
    <br />
    <a href="https://github.com/othneildrew/Best-README-Template"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/othneildrew/Best-README-Template">View Demo</a>
    &middot;
    <a href="https://github.com/othneildrew/Best-README-Template/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    &middot;
    <a href="https://github.com/othneildrew/Best-README-Template/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
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

I highly recommend using micromamba to setup Deepen's environment. Installing micromamba is super simple (instructions here: (https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html). Currently, Deepen's GPU backend only works on Linux/Windows (you must be able to use ```CuPy```). The compiler backend is GPU-only (uses NVIDIA's CUDA libraries and CUDA C/C++ code directly).

```
micromamba create -n deepen -c conda-forge python=3.13 numpy cupy ipykernel
```

You can pick between installing either ```NumPy``` or ```CuPy```. The last package, ```ipykernel``` is generally useful as it lets you run code in Jupyter-style notebooks in VSCode/Cursor. Eventually, I'll update this to include instructions on how to setup ```rust```, ```cuda```, and their related packages once the compiler is ready.

## Examples



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
