<!-- PROJECT TITLE/LOGO -->
<br />
<div align="center">
  <!-- <a href="https://github.com/Catastrophe9260/deepen">
    <img src="logo.png" alt="Logo" width="80" height="80">
  </a> -->

  <h1 align="center">Deepen</h1>

  <p align="center">
    A compilation-enabled, static-graph deep learning framework.
  </p>
</div>

<!-- ABOUT THE PROJECT -->
## About

Deepen is a from-scratch, static-graph deep learning framework that implements automatic differentiation, computation graphs, and a complete suite of neural network primitives. Inspired mainly by PyTorch and a bit by JAX, it provides a flexible, abstracted tensor interface with dual execution modes - lazy graph building by default and eager execution for debugging - along with a full reverse-mode autodiff system for training neural networks. It also has a pluggable backend system supporting both NumPy (CPU) and CuPy (GPU), comprehensive neural network layers (Linear, Conv2d, BatchNorm, etc), popular optimizers (SGD, RMSprop, Adam, etc), and common loss functions (MSE, BCE, etc).

I'm currently working on a Rust-based compiler (DeepX) with graph serialization to an SSA-based IR for model compilation. The compiler is primarily focused on implementing graph-level optimization passes (DCE, CSE, etc) and operator fusion. This system routes to a custom CUDA backend built using cuBLAS/cuDNN and fused CUDA kernels.

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
compose.py --> sequential container, activation wrappers, model blocks (residuals, support for CNNs, Transformers, etc)
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

I highly recommend using `micromamba` to setup Deepen's environment. Installing `micromamba` is super simple ([instructions here](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html)). Currently, Deepen's GPU backend only works on Linux/Windows (you must be able to use `CuPy`). The compiler backend is GPU-only (uses `cuBLAS`/`cuDNN` and `CUDA` code directly).

```
micromamba create -n deepen_env -c conda-forge python=3.13 numpy cupy ipykernel
```

Once you've done that, just clone this repo to any directory you want, activate your `micromamba` environment (e.g. deepen_env, if you used the command from before), and start exploring! Further down below is an example you can copy and run immediately (paste it into a `.py` or `.ipynb` file outside deepen/). If you're using a Jupyter notebook in VSCode/Cursor, make sure your `micromamba` environment is recognized and selected.

Note that, you can pick between installing either `NumPy` or `CuPy`. The last package, `ipykernel` is generally useful as it lets you run code in Jupyter-style notebooks in VSCode/Cursor. Eventually, I'll update this to include instructions on how to setup `rust`, `cuda-toolkit`, and their related packages with `micromamba` (once the compiler is ready).

Deepen's goal is to be very approachable. While the lack of documentation (hopefully, not for long) makes that seem conterintuitive, the code itself is in `Python` and super readable. If you want to understand how a deep learning framework is structured - my hope is that exploring Deepen will be much easier than diving into PyTorch's source code.

With that mindset, I've also decided to build a compiler for it in `Rust`. The goal is the same - write super readable systems code to show you the essence of how a deep learning compiler works (you might lose that guarantee from me for `CUDA` code, however). Other projects like this are usually too simple and don't teach you much. Deepen is fully-featured, supports a near-complete subset of deep learning models, and you get to see someone design and build a compiler for it in real-time.

Of course, Deepen is a gentle introduction to deep learning systems and hides a lot of complexities you'd usually find in frameworks like PyTorch, or worse, JAX (more of a differentiable programming framework, but similar in capability). While building it, I used, and still use a super cool tool called *DeepWiki* (based on [Devin](devin.ai)). It first indexes large codebases with an LLM after which you can ask it questions of any kind. You can find [PyTorch](https://deepwiki.com/pytorch/pytorch) and [JAX](https://deepwiki.com/jax-ml/jax) already indexed. They're pretty frequently updated too!

## Example

Below is a pretty simple example of using Deepen to train a small neural network on simulated data that follows a sine curve. I have an MNIST example that uses a CNN in the works.

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

## Acknowledgements

One of the reasons I wanted to build a project like this was inspiration from talking to Dr. Tianqi Chen - my professor at CMU. He teaches two amazing courses that I've taken: *10-414: Deep Learning Systems: Algorithms and Implementation* and *15-442: Machine Learning Systems* which together cover a variety of material spanning automatic differentiation, neural network library abstractions, hardware acceleration with GPU programming (GPU memory models, low-level CUDA kernel implementations, etc), things like TVM (an ML-based deep learning compiler), efficient deep learning model implementations (optimizing how convolutions are implemented under-the-hood, etc), and other advanced stuff that changes year-to-year.

The primary reason, however, was that my research at CMU lies at the intersection of neuroscience and deep learning - and while I was good at using libraries like PyTorch (for coursework and personal projects) and JAX (for my research) - I wanted to really learn how these things worked internally. Turns out, the best way to do that, is to build one yourself.

<!-- LICENSE -->
## License

MIT License - see [LICENSE](LICENSE)
