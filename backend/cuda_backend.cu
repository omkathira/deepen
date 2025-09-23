#include <math.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

// Element-wise addition
template <typename T>
__global__ void add_TT(T const* __restrict__ x, T const* __restrict__ y, T* __restrict__ output, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x; // from 0 to N - 1
    int stride = blockDim.x * gridDim.x;
    for (; i < N; i += stride) {
        output[i] = x[i] + y[i];
    } // end for loop
} // end add_TT

template <typename T>
__global__ void add_TS(T const* __restrict__ x, T y, T* __restrict__ output, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (; i < N; i += stride) {
        output[i] = x[i] + y;
    } // end for loop
} // end add_TS

// Element-wise subtraction
template <typename T>
__global__ void sub_TT(T const* __restrict__ x, T const* __restrict__ y, T* __restrict__ output, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (; i < N; i += stride) {
        output[i] = x[i] - y[i];
    } // end for loop
} // end sub_TT

template <typename T>
__global__ void sub_TS(T const* __restrict__ x, T y, T* __restrict__ output, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (; i < N; i += stride) {
        output[i] = x[i] - y;
    } // end for loop
} // end sub_TS

template <typename T>
__global__ void sub_ST(T const* __restrict__ x, T y, T* __restrict__ output, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (; i < N; i += stride) {
        output[i] = y - x[i];
    } // end for loop
} // end sub_ST

// Element-wise multiplication
template <typename T>
__global__ void mul_TT(T const* __restrict__ x, T const* __restrict__ y, T* __restrict__ output, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (; i < N; i += stride) {
        output[i] = x[i] * y[i];
    } // end for loop
} // end mul_TT

template <typename T>
__global__ void mul_TS(T const* __restrict__ x, T y, T* __restrict__ output, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (; i < N; i += stride) {
        output[i] = x[i] * y;
    } // end for loop
} // end mul_TS

// Element-wise division
template <typename T>
__global__ void div_TT(T const* __restrict__ x, T const* __restrict__ y, T* __restrict__ output, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (; i < N; i += stride) {
        output[i] = x[i] / y[i];
    } // end for loop
} // end div_TT

template <typename T>
__global__ void div_TS(T const* __restrict__ x, T y, T* __restrict__ output, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (; i < N; i += stride) {
        output[i] = x[i] / y;
    } // end for loop
} // end div_TS

template <typename T>
__global__ void div_ST(T const* __restrict__ x, T y, T* __restrict__ output, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (; i < N; i += stride) {
        output[i] = y / x[i];
    } // end for loop
} // end div_ST

// Negation
template <typename T>
__global__ void neg_T(T const* __restrict__ x, T* __restrict__ output, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (; i < N; i += stride) {
        output[i] = -x[i]; // applying x unary minus is type-safe
    } // end for loop
} // end neg

// Absolute value
__device__ inline float abs_t(float x) {return fabsf(x);}
__device__ inline double abs_t(double x) {return fabs(x);}

template <typename T>
__global__ void abs_T(T const* __restrict__ x, T* __restrict__ output, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (; i < N; i += stride) {
        output[i] = abs_t(x[i]);
    } // end for loop
} // end abs

// Power
__device__ inline float pow_t(float x, float y) {return powf(x, y);}
__device__ inline double pow_t(double x, double y) {return pow(x, y);}

template <typename T>
__global__ void pow_TT(T const* __restrict__ x, T const* __restrict__ y, T* __restrict__ output, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (; i < N; i += stride) {
        output[i] = pow_t(x[i], y[i]);
    } // end for loop
} // end pow_TT

template <typename T>
__global__ void pow_TS(T const* __restrict__ x, T n, T* __restrict__ output, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (; i < N; i += stride) {
        output[i] = pow_t(x[i], n);
    } // end for loop
} // end pow_TS

// Exponentiation
__device__ inline float exp_t(float x) {return expf(x);}
__device__ inline double exp_t(double x) {return exp(x);}

template <typename T>
__global__ void exp_T(T const* __restrict__ x, T* __restrict__ output, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (; i < N; i += stride) {
        output[i] = exp_t(x[i]);
    } // end for loop
} // end exp_T

// Logarithm
__device__ inline float log_t(float x) {return logf(x);}
__device__ inline double log_t(double x) {return log(x);}

template <typename T>
__global__ void log_T(T const* __restrict__ x, T* __restrict__ output, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (; i < N; i += stride) {
        output[i] = log_t(x[i]);
    } // end for loop
} // end log_T

// Clip
__device__ inline float min_t(float x, float y) {return fminf(x, y);}
__device__ inline double min_t(double x, double y){ return fmin(x, y);}

__device__ inline float max_t(float x, float y) {return fmaxf(x, y);}
__device__ inline double max_t(double x, double y) {return fmax(x, y);}

template <typename T>
__global__ void clip_T(T const* __restrict__ x, T min_val, T max_val, T* __restrict__ output, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (; i < N; i += stride) {
        output[i] = max_t(min_val, min_t(max_val, x[i]));
    } // end for loop
} // end clip_T

// Sign
template <typename T>
__global__ void sign_T(T const* __restrict__ x, T* __restrict__ output, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (; i < N; i += stride) {
        output[i] = (x[i] > 0) ? 1 : ((x[i] < 0) ? -1 : 0);
    } // end for loop
} // end sign_T

