import os
import cupy as cp

with open(os.path.join(os.path.dirname(__file__), 'kernels', 'add_relu.cu'), 'r') as f:
    add_relu_kernel = f.read()

_add_relu_kernel = cp.RawKernel(add_relu_kernel, 'add_relu')

def add_relu(x, y):
    output = cp.empty_like(x)
    size = x.size
    threads_per_block = 256
    blocks = (size + threads_per_block - 1) // threads_per_block

    _add_relu_kernel(
        (blocks,), (threads_per_block,),
        (x, y, output, size)
    )

    return output

with open(os.path.join(os.path.dirname(__file__), 'kernels', 'mul_relu.cu'), 'r') as f:
    mul_relu_kernel = f.read()

_mul_relu_kernel = cp.RawKernel(mul_relu_kernel, 'mul_relu')

def mul_relu(x, y):
    output = cp.empty_like(x)
    size = x.size
    threads_per_block = 256
    blocks = (size + threads_per_block - 1) // threads_per_block

    _mul_relu_kernel(
        (blocks,), (threads_per_block,),
        (x, y, output, size)
    )

    return output