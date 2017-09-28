#ifndef __PHYSICS_MODEL_H__
#define __PHYSICS_MODEL_H__

#include "cuda.h"
#include "cuda_runtime.h"

class Physics
{
public:

	/*
		Calculate the epsilon matrix (Strain)
	*/
	__host__ __device__
	static void CalculateEpsilon(float eps[9], float a[4], float b[4], float c[4]);

	/*
		Calculate the mechanical potential energy using strain
	*/
	__host__ __device__
	static void CalculateElasticPotential(float &localPe, const float eps[9], const float &cxxxx, const float &cxxyy, const float &cxyxy);
	
	/*
		Calculate energy from liquid crystal order
	*/
	__host__ __device__
	static void CalculateLiquidCrystalEnergy(float &lcEnergy, const float eps[9], const float Q[9], const float &alpha);
	
	/*
		Forces from elastic energy
	*/
	__host__ __device__
	static void AddElasticForces(float F[12], const float eps[9], const float Ainv[16], const float a[4], const float b[4], const float c[4], const float &cxxxx, const float &cxxyy, const float &cxyxy);

	/*
		Forces from liquid crystal 
	*/
	__host__ __device__
	static void AddLiquidCrystalForces(float F[12], const float Q[9], const float Ainv[16], const float a[4], const float b[2], const float c[4], const float &alpha);

	/*
		Calculate Shape Function
	*/
	__host__ __device__
	static void CalculateShapeFunction(float a[4], float b[4], float c[4], const float r[12], const float r0[12], const float Ainv[16]);
};

#endif
