import triton
import triton.language as tl
import cupy as cp
import numpy as np

class CuPyPtr:
    def __init__(self, cupy_array):
        self.ptr = cupy_array.__cuda_array_interface__['data'][0]
        self.dtype = cupy_array.dtype
    
    def data_ptr(self):
        return self.ptr

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

if __name__ == "__main__":
    # Create CuPy arrays on GPU
    x_cupy = cp.arange(1024, dtype=cp.float32)
    y_cupy = cp.arange(1024, dtype=cp.float32)
    output_cupy = cp.empty(1024, dtype=cp.float32)
    
    # Wrap CuPy arrays with pointer wrapper
    x = CuPyPtr(x_cupy)
    y = CuPyPtr(y_cupy)
    output = CuPyPtr(output_cupy)
    
    # Launch kernel
    grid = lambda meta: (triton.cdiv(1024, meta['BLOCK_SIZE']),)
    add_kernel[grid](x, y, output, 1024, BLOCK_SIZE=256)
    
    # Verify
    expected = x_cupy + y_cupy
    print(f"Match: {cp.allclose(output_cupy, expected)}")
    print(f"First 10: {output_cupy[:10]}")