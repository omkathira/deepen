import cupy as cp

add_relu_kernel = r'''
extern "C" __global__
void add_relu(const float* x, const float* y, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float result = x[idx] + y[idx];
        out[idx] = result > 0.0f ? result : 0.0f;
    }
}
'''

_add_relu_kernel = cp.RawKernel(add_relu_kernel, 'add_relu')

def add_relu(x, y):
    assert x.shape == y.shape
    assert x.dtype == cp.float32

    out = cp.empty_like(x)
    size = x.size
    threads_per_block = 256
    blocks = (size + threads_per_block - 1) // threads_per_block

    _add_relu_kernel(
        (blocks,), (threads_per_block,),
        (x, y, out, size)
    )

    return out