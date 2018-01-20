
#include "../include/physics_model.h"
#include "cuda_runtime.h"
#include "../include/defines.h"

__host__ __device__
void Physics::CalculateEpsilon(
	real eps[9], 
	real a[4], 
	real b[4], 
	real c[4])
{
	eps[3*0+0] = a[1]+0.5*(a[1]*a[1]+b[1]*b[1]+c[1]*c[1]);
	eps[3*1+1] = b[2]+0.5*(a[2]*a[2]+b[2]*b[2]+c[2]*c[2]);
	eps[3*2+2] = c[3]+0.5*(a[3]*a[3]+b[3]*b[3]+c[3]*c[3]);
	eps[3*0+1] = 0.5*(a[2]+b[1]+a[1]*a[2]+b[1]*b[2]+c[1]*c[2]);
	eps[3*1+0] = eps[3*0+1];
	eps[3*0+2] = 0.5*(a[3]+c[1]+a[1]*a[3]+b[1]*b[3]+c[1]*c[3]);
	eps[3*2+0] = eps[3*0+2];
	eps[3*1+2] = 0.5*(b[3]+c[2]+a[2]*a[3]+b[2]*b[3]+c[2]*c[3]);
	eps[3*2+1] = eps[3*1+2];
}


__host__ __device__
void Physics::CalculateLiquidCrystalEnergy(
	real &lcEnergy, 
	const real eps[9], 
	const real Q[9], 
	const real &alpha)
{
	// alpha has units of energy density 
	lcEnergy = -1.0*alpha*(eps[3*0+0]*Q[3*0+0]
				+eps[3*1+1]*Q[3*1+1]+eps[3*2+2]*Q[3*2+2]
				+eps[3*0+1]*Q[3*0+1]+eps[3*1+0]*Q[3*1+0]
				+eps[3*1+2]*Q[3*1+2]+eps[3*2+1]*Q[3*2+1]
				+eps[3*0+2]*Q[3*0+2]+eps[3*2+0]*Q[3*2+0]);
}


__host__ __device__
void Physics::CalculateElasticPotential(
	real &local_Pe, 
	const real eps[9], 
	const real &cxxxx, 
	const real &cxxyy, 
	const real &cxyxy)
{
	local_Pe = 0.0f;
	local_Pe += cxxxx*(eps[3*0+0]*eps[3*0+0]+eps[3*1+1]*eps[3*1+1]+eps[3*2+2]*eps[3*2+2]);
	local_Pe += 2.0*cxxyy*(eps[3*0+0]*eps[3*1+1]+eps[3*1+1]*eps[3*2+2]+eps[3*0+0]*eps[3*2+2]);
	local_Pe += 4.0*cxyxy*(eps[3*0+1]*eps[3*0+1]+eps[3*1+2]*eps[3*1+2]+eps[3*2+0]*eps[3*2+0]);
}

__host__ __device__
void Physics::AddElasticForces(
	real F[12], 
	const real eps[9], 
	const real Ainv[16], 
	const real a[4], 
	const real b[4], 
	const real c[4], 
	const real &cxxxx, 
	const real &cxxyy, 
	const real &cxyxy)
{
	for(int n = 0; n < 4; n++)
	{
		//x
		F[0+n*3]+=-cxxxx*2.0f*eps[3*0+0]*Ainv[4*1+n]*(1.0f+a[1]);
		F[0+n*3]+=-cxxxx*2.0f*eps[3*1+1]*Ainv[4*2+n]*a[2];
		F[0+n*3]+=-cxxxx*2.0f*eps[3*2+2]*Ainv[4*3+n]*a[3];
		F[0+n*3]+=-2.0f*cxxyy*eps[3*0+0]*Ainv[4*2+n]*a[2];
		F[0+n*3]+=-2.0f*cxxyy*eps[3*1+1]*Ainv[4*1+n]*(1.0f+a[1]);
		F[0+n*3]+=-2.0f*cxxyy*eps[3*1+1]*Ainv[4*3+n]*a[3];
		F[0+n*3]+=-2.0f*cxxyy*eps[3*2+2]*Ainv[4*2+n]*a[2];
		F[0+n*3]+=-2.0f*cxxyy*eps[3*2+2]*Ainv[4*1+n]*(1.0f+a[1]);
		F[0+n*3]+=-2.0f*cxxyy*eps[3*0+0]*Ainv[4*3+n]*a[3];
		F[0+n*3]+=-4.0f*cxyxy*eps[3*0+1]*Ainv[4*2+n]*(1.0f+a[1]);
		F[0+n*3]+=-4.0f*cxyxy*eps[3*0+1]*Ainv[4*1+n]*a[2];
		F[0+n*3]+=-4.0f*cxyxy*eps[3*1+2]*Ainv[4*3+n]*a[2];
		F[0+n*3]+=-4.0f*cxyxy*eps[3*1+2]*Ainv[4*2+n]*a[3];
		F[0+n*3]+=-4.0f*cxyxy*eps[3*2+0]*Ainv[4*3+n]*(1.0f+a[1]);
		F[0+n*3]+=-4.0f*cxyxy*eps[3*2+0]*Ainv[4*1+n]*a[3];

		//y
		F[1+n*3]+=-cxxxx*2.0f*eps[3*0+0]*Ainv[4*1+n]*b[1];
		F[1+n*3]+=-cxxxx*2.0f*eps[3*1+1]*Ainv[4*2+n]*(1.0f+b[2]);
		F[1+n*3]+=-cxxxx*2.0f*eps[3*2+2]*Ainv[4*3+n]*b[3];
		F[1+n*3]+=-2.0f*cxxyy*eps[3*0+0]*Ainv[4*2+n]*(1.0f+b[2]);
		F[1+n*3]+=-2.0f*cxxyy*eps[3*1+1]*Ainv[4*1+n]*b[1];
		F[1+n*3]+=-2.0f*cxxyy*eps[3*1+1]*Ainv[4*3+n]*b[3];
		F[1+n*3]+=-2.0f*cxxyy*eps[3*2+2]*Ainv[4*2+n]*(1.0f+b[2]);
		F[1+n*3]+=-2.0f*cxxyy*eps[3*2+2]*Ainv[4*1+n]*b[1];
		F[1+n*3]+=-2.0f*cxxyy*eps[3*0+0]*Ainv[4*3+n]*b[3];
		F[1+n*3]+=-4.0f*cxyxy*eps[3*0+1]*Ainv[4*1+n]*(1.0f+b[2]);
		F[1+n*3]+=-4.0f*cxyxy*eps[3*0+1]*Ainv[4*2+n]*b[1];
		F[1+n*3]+=-4.0f*cxyxy*eps[3*1+2]*Ainv[4*3+n]*(1.0f+b[2]);
		F[1+n*3]+=-4.0f*cxyxy*eps[3*1+2]*Ainv[4*2+n]*b[3];
		F[1+n*3]+=-4.0f*cxyxy*eps[3*2+0]*Ainv[4*3+n]*b[1];
		F[1+n*3]+=-4.0f*cxyxy*eps[3*2+0]*Ainv[4*1+n]*b[3];

		//z
		F[2+n*3]+=-cxxxx*2.0f*eps[3*0+0]*Ainv[4*1+n]*c[1];
		F[2+n*3]+=-cxxxx*2.0f*eps[3*1+1]*Ainv[4*2+n]*c[2];
		F[2+n*3]+=-cxxxx*2.0f*eps[3*2+2]*Ainv[4*3+n]*(1.0f+c[3]);
		F[2+n*3]+=-2.0f*cxxyy*eps[3*0+0]*Ainv[4*2+n]*c[2];
		F[2+n*3]+=-2.0f*cxxyy*eps[3*1+1]*Ainv[4*1+n]*c[1];
		F[2+n*3]+=-2.0f*cxxyy*eps[3*1+1]*Ainv[4*3+n]*(1.0f+c[3]);
		F[2+n*3]+=-2.0f*cxxyy*eps[3*2+2]*Ainv[4*2+n]*c[2];
		F[2+n*3]+=-2.0f*cxxyy*eps[3*2+2]*Ainv[4*1+n]*c[1];
		F[2+n*3]+=-2.0f*cxxyy*eps[3*0+0]*Ainv[4*3+n]*(1.0f+c[3]);
		F[2+n*3]+=-4.0f*cxyxy*eps[3*0+1]*Ainv[4*2+n]*c[1];
		F[2+n*3]+=-4.0f*cxyxy*eps[3*0+1]*Ainv[4*1+n]*c[2];
		F[2+n*3]+=-4.0f*cxyxy*eps[3*1+2]*Ainv[4*2+n]*(1.0f+c[3]);
		F[2+n*3]+=-4.0f*cxyxy*eps[3*1+2]*Ainv[4*3+n]*c[2];
		F[2+n*3]+=-4.0f*cxyxy*eps[3*2+0]*Ainv[4*1+n]*(1.0f+c[3]);
		F[2+n*3]+=-4.0f*cxyxy*eps[3*2+0]*Ainv[4*3+n]*c[1];
	}
}


__host__ __device__
void Physics::AddLiquidCrystalForces(
	real F[12], 
	const real Q[9], 
	const real Ainv[16], 
	const real a[4], 
	const real b[2], 
	const real c[4], 
	const real &alpha)
{
	for(int n = 0; n < 4; n++)
	{
		F[0+n*3] += alpha * (Q[0*3+0] * Ainv[4*1+n] * (1.0f + a[1])
				  +Q[1*3+1] * Ainv[4*2+n] * a[2]
				  +Q[2*3+2] * Ainv[4*3+n] * a[3]
				  +Q[0*3+1] * (Ainv[4*2+n] * (1.0f + a[1]) + Ainv[4*1+n] * a[2])
				  +Q[0*3+2] * (Ainv[4*3+n] * (1.0f + a[1]) + Ainv[4*1+n] * a[3])
				  +Q[1*3+2] * (Ainv[4*3+n] * a[2]+Ainv[4*2+n] * a[3]));

		F[1+n*3] += alpha * (Q[0*3+0] * Ainv[4*1+n] * b[1]
				  +Q[1*3+1] * Ainv[4*2+n] * (1.0f + b[2])
				  +Q[2*3+2] * Ainv[4*3+n] * b[3]
				  +Q[0*3+1] * (Ainv[4*1+n] * (1.0f + b[2]) + Ainv[4*2+n] * b[1])
				  +Q[0*3+2] * (Ainv[4*3+n] * b[1] + Ainv[4*1+n] * b[3])
				  +Q[1*3+2] * (Ainv[4*3+n] * (1.0f + b[2]) + Ainv[4*2+n] * b[3]));

		F[2+n*3] += alpha * (Q[0*3+0] * Ainv[4*1+n] * c[1]
				  +Q[1*3+1] * Ainv[4*2+n] * c[2]
				  +Q[2*3+2] * Ainv[4*3+n] * (1.0f + c[3])
				  +Q[0*3+1] * (Ainv[4*2+n] * c[1] + Ainv[4*1+n] * c[2])
				  +Q[0*3+2] * (Ainv[4*1+n] * (1.0f + c[3]) + Ainv[4*3+n] * c[1])
				  +Q[1*3+2] * (Ainv[4*2+n] * (1.0f + c[3]) + Ainv[4*3+n] * c[2]));
	}
}



__host__ __device__
void Physics::CalculateShapeFunction(
	real a[4], 
	real b[4], 
	real c[4], 
	const real r[12], 
	const real r0[12], 
	const real Ainv[16])
{
	real u[4], v[4] ,w[4];
	a[0] = a[1] = a[2] = a[3] = 0.0f;
	b[0] = b[1] = b[2] = b[3] = 0.0f;
	c[0] = c[1] = c[2] = c[3] = 0.0f;
	
	//clacluate displacements from original position and zero out forces
	for(int n = 0; n < 4; n++){
		u[n] = (r[n*3] - r0[n*3]);
		v[n] = (r[1+n*3] - r0[1+n*3]);
		w[n] = (r[2+n*3] - r0[2+n*3]);
	}//n

	//matrix multipy Ainv and u,v,w to get shape funcitons
	for(int i = 0; i < 4; i++){
		for(int j = 0; j < 4; j++){
			a[i] += Ainv[4*i+j] * u[j];
			b[i] += Ainv[4*i+j] * v[j];
			c[i] += Ainv[4*i+j] * w[j];
		}//j
	}//i
}

//=============================================================
//calculate forces on all 4 nodes in tetrahedra
//=============================================================
__device__ 
void Physics::CalculateForcesAndEnergies(PackedParameters params,
	real *Ainv,
	real *r0,
	real *r,
	real *Q,
	real (&F)[12],
	int *TetNodeRank,
	real *pe,
	int mytet,
	real myVol)
{

	//real u[4],v[4],w[4];
	real eps[9];
	real a[4], b[4], c[4];
	real localPe = 0.0;
	real lcEnergy = 0.0f;
	const real cxxxx = params.Cxxxx;
	const real cxxyy = params.Cxxyy;
	const real cxyxy = params.Cxyxy;
	const real alpha = params.Alpha;

	Physics::CalculateShapeFunction(a, b, c, r, r0, Ainv);
	Physics::CalculateEpsilon(eps, a, b, c);
	Physics::CalculateElasticPotential(localPe, eps, cxxxx, cxxyy, cxyxy);
	Physics::AddElasticForces(F, eps, Ainv, a, b, c, cxxxx, cxxyy, cxyxy);
	Physics::AddLiquidCrystalForces(F, Q, Ainv, a, b, c, alpha);
	Physics::CalculateLiquidCrystalEnergy(lcEnergy, eps, Q, alpha);
	pe[mytet] = (localPe + lcEnergy) * myVol;
}

