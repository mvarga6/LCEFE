#ifndef __PHYSICS_KERNELS_H__
#define __PHYSICS_KERNELS_H__

#include "cuda.h"
#include "cuda_runtime.h"
#include "datastruct.h"


//================================================
// -calculate all forces and energies inside each tet
// -write forces to appropiate location in node
//	 force container
//================================================
__global__ void ForceKernel(DevDataBlock data, real t);

//================================================
// -read in all forces on each node and sum them
// -use velocity verilet to step system forward
// -send updated positions to global memory 
//================================================
__global__ void UpdateKernel(DevDataBlock data);

#endif
