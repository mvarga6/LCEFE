#ifndef __PHYSICS_KERNELS_H__
#define __PHYSICS_KERNELS_H__

#include "cuda.h"
#include "cuda_runtime.h"
#include "datastruct.h"


///================================================
/// -calculate all forces and energies inside each tet
/// -write forces to appropiate location in node
///	 force container
///================================================
__global__ void BulkForceKernel(DevDataBlock data, real t);


///================================================
/// -calculate the enclosed volumes within given data block
///================================================
__global__ void CalculateClosedVolumesKernel(DevDataBlock data, float3 center);

///================================================
/// -calculate all forces on surfaces from pressure
///  given an initial volume and a current volume
///================================================
__global__ void PressureForcesKernel(DevDataBlock data, const real V, const real V0, const real Kp);

///================================================
/// -read in all forces on each node and sum them
/// -use velocity verilet to step system forward
/// -send updated positions to global memory 
///================================================
__global__ void UpdateKernel(DevDataBlock data);

#endif
