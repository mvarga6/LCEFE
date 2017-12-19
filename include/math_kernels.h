#ifndef __MATH_KERNELS_H__
#define __MATH_KERNELS_H__

#include "cuda.h"
#include "cuda_runtime.h"
#include "defines.h"

///
/// Kernels to add to global memory in many different ways
///

__global__ void Add(real *ptr, const real value, int N);
__global__ void Add(int *ptr, const int value, int N);
__global__ void Add(real *ptr, real *values, int N);
//__global__ void ReduceSum(real * __restrict__ input, real * __restrict__ reduced);

#endif