#ifndef __PHYSICS_MODEL_H__
#define __PHYSICS_MODEL_H__

#include "cuda.h"
#include "cuda_runtime.h"
#include "defines.h"
#include "simulation_parameters.h"

class Physics
{
public:

	/*
		Calculate the epsilon matrix (Strain)
	*/
	__host__ __device__
	static void CalculateEpsilon(real eps[9], real a[4], real b[4], real c[4]);

	/*
		Calculate the mechanical potential energy using strain
	*/
	__host__ __device__
	static void CalculateElasticPotential(real &local_Pe, const real eps[9], const real &cxxxx, const real &cxxyy, const real &cxyxy);
	
	/*
		Calculate energy from liquid crystal order
	*/
	__host__ __device__
	static void CalculateLiquidCrystalEnergy(real &lcEnergy, const real eps[9], const real Q[9], const real &alpha);
	
	/*
		Forces from elastic energy
	*/
	__host__ __device__
	static void AddElasticForces(real F[12], const real eps[9], const real Ainv[16], const real a[4], const real b[4], const real c[4], const real &cxxxx, const real &cxxyy, const real &cxyxy);

	/*
		Forces from liquid crystal 
	*/
	__host__ __device__
	static void AddLiquidCrystalForces(real F[12], const real Q[9], const real Ainv[16], const real a[4], const real b[2], const real c[4], const real &alpha);

	/*
		Calculate Shape Function
	*/
	__host__ __device__
	static void CalculateShapeFunction(real a[4], real b[4], real c[4], const real r[12], const real r0[12], const real Ainv[16]);

	/*
		Calculates all physics Forces
	*/
	__device__
	static void CalculateForcesAndEnergies(PackedParameters params, real *Ainv,real *r0,real *r,real *Q,real (&F)[12],int *TetNodeRank,real *pe,int mytet,real myVol);
};

#endif
