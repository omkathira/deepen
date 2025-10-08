extern "C" __global__
void add_relu(const float* x, const float* y, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float result = x[idx] + y[idx];
        out[idx] = result > 0.0f ? result : 0.0f;
    }
}