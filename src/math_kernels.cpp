#include "math_kernels.h"

__global__ void Add(real *ptr, const real value, int N)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N)
    {
        ptr[tid] += value;
    }
}

__global__ void Add(int *ptr, const int value, int N)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N)
    {
        ptr[tid] += value;
    }
}

__global__ void Add(real *ptr, real *values, int N)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N)
    {
        ptr[tid] += values[tid];
    }
}