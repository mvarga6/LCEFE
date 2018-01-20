#include "../include/math_kernels.h"
//#include "cub.cuh"

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

// __global__ void ReduceSum(real * __restrict__ input, real * __restrict__ reduced, const int N)
// {
//     unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
//     // --- Specialize BlockReduce for type float. 
//     typedef cub::BlockReduce<real, blockDim.x> BlockReduceT; 

//     // --- Allocate temporary storage in shared memory 
//     __shared__ typename BlockReduceT::TempStorage temp_storage; 

//     real result;
//     if (tid < N) result = BlockReduceT(temp_storage).Sum(input[tid]);

//     // --- Update block reduction value
//     if (threadIdx.x == 0) reduced[blockIdx.x] = result;
// }