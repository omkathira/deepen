#include <stdio.h>
#include <iostream>

// #define N 345

// __global__ void add(int *a, int *b, int *c) {
//     int i = threadIdx.x + blockIdx.x * blockDim.x; // from 0 to N - 1
//     while (i < N) {
//         c[i] = a[i] + b[i];
//         i += blockDim.x + gridDim.x;
//     }
// }

#define imin(a, b) (a < b ? a : b) // if a < b return a, otherwise b

const int N = 33 * 1024;
const int threadsPerBlock = 256;
const int blocksPerGrid = imin(32, (N + threadsPerBlock - 1) / threadsPerBlock);

__global__ void dot(float *a, float *b, float *c) {
    __shared__ float cache[threadsPerBlock];

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIdx = threadIdx.x; // cache size = number of threads

    float temp = 0;
    while (i < N) {
        temp += a[i] * b[i]; // stores a running sum per thread
        i += blockDim.x * gridDim.x;
    }

    cache[cacheIdx] = temp; // store multiplications in shared memory

    __syncthreads();

    int offset = blockDim.x / 2;

    while (offset != 0) {
        if (cacheIdx < offset) {
            cache[cacheIdx] += cache[cacheIdx + offset];
        }

        __syncthreads(); // make sure active threads finished and wrote to cache
        offset /= 2;
    }

    if (cacheIdx == 0) {
        c[blockIdx.x] = cache[0]; // collect dot product result from each block
    }
}

int main(void) {

    float *a, *b, c, *p_c;
    float *dev_a, *dev_b, *dev_p_c;

    a = new float[N];
    b = new float[N];
    p_c = new float[blocksPerGrid];

    cudaMalloc((void**)&dev_a, N * sizeof(float));
    cudaMalloc((void**)&dev_b, N * sizeof(float));
    cudaMalloc((void**)&dev_p_c, blocksPerGrid * sizeof(float));

    for (int i = 0; i < N; i++) {
        a[i] = i;
        b[i] = i * 2;
    }

    cudaMemcpy(dev_a, a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, N * sizeof(float), cudaMemcpyHostToDevice);

    dot<<<blocksPerGrid,threadsPerBlock>>>(dev_a, dev_b, dev_p_c);

    cudaMemcpy(p_c, dev_p_c, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);

    c = 0;
    for (int i = 0; i < blocksPerGrid; i++) {
        c += p_c[i];
    }

    #define sum_squares(x) (x*(x+1)*(2*x+1)/6)
    printf("Does GPU value %.6g = %.6g?\n", c, 2 * sum_squares((float)(N - 1)));

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_p_c);

    delete [] a;
    delete [] b;
    delete [] p_c;

// ----------------------------------------------------------

    // int a[N], b[N], c[N];
    // int *dev_a, *dev_b, *dev_c;

    // cudaMalloc((void**)&dev_a, N * sizeof(int));
    // cudaMalloc((void**)&dev_b, N * sizeof(int));
    // cudaMalloc((void**)&dev_c, N * sizeof(int));

    // for(int i = 0; i < N; i++) {
    //     a[i] = -i;
    //     b[i] = i * i;
    // }

    // cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
    // cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);

    // // first number is number of parallel blocks to execute program in
    // // second number is the number of threads per block
    // add<<<128,128>>>(dev_a, dev_b, dev_c);

    // cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost);

    // cudaFree(dev_a);
    // cudaFree(dev_b);
    // cudaFree(dev_c);

    // for (int i=0; i<N; i++) {
    //     printf("%d + %d = %d\n", a[i], b[i], c[i]);
    // }

// ----------------------------------------------------------

    // int c;
    // int *dev_c;

    // cudaMalloc((void**)&dev_c, sizeof(int));

    // add<<<1,1>>>(2, 7, dev_c);
    // cudaDeviceSynchronize(); // prevent host from running before device finishes calculation

    // cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost);

    // printf("Hello world!\n");
    // printf("2 + 7 = %d\n", c);

    // cudaFree(dev_c);


    // cudaDeviceProp prop;

    // int count;

    // cudaGetDeviceCount(&count);

    // for (int i = 0; i < count; i++) {
    //     cudaGetDeviceProperties(&prop, i);
    
    //     printf("name: %s\n", prop.name);
    //     printf("mem: %zu bytes\n", prop.totalGlobalMem);
    // }

// ----------------------------------------------------------

    return 0;
}