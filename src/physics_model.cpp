
#include "physics_model.h"
#include "cuda_runtime.h"


__host__ __device__
void Physics::CalculateEpsilon(
	float eps[9], 
	float a[4], 
	float b[4], 
	float c[4])
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


__device__
void CalculateElasticPotential(
	float &local_Pe, 
	const float eps[9], 
	const float &cxxxx, 
	const float &cxxyy, 
	const float &cxyxy)
{
	local_Pe = 0.0f;
	local_Pe += cxxxx*(eps[3*0+0]*eps[3*0+0]+eps[3*1+1]*eps[3*1+1]+eps[3*2+2]*eps[3*2+2]);
	local_Pe += 2.0*cxxyy*(eps[3*0+0]*eps[3*1+1]+eps[3*1+1]*eps[3*2+2]+eps[3*0+0]*eps[3*2+2]);
	local_Pe += 4.0*cxyxy*(eps[3*0+1]*eps[3*0+1]+eps[3*1+2]*eps[3*1+2]+eps[3*2+0]*eps[3*2+0]);
}


__host__ __device__
void Physics::CalculateLiquidCrystalEnergy(
	float &lcEnergy, 
	const float eps[9], 
	const float Q[9], 
	const float &alpha)
{
	// alpha has units of energy density 
	lcEnergy = -1.0*alpha*(eps[3*0+0]*Q[3*0+0]
				+eps[3*1+1]*Q[3*1+1]+eps[3*2+2]*Q[3*2+2]
				+eps[3*0+1]*Q[3*0+1]+eps[3*1+0]*Q[3*1+0]
				+eps[3*1+2]*Q[3*1+2]+eps[3*2+1]*Q[3*2+1]
				+eps[3*0+2]*Q[3*0+2]+eps[3*2+0]*Q[3*2+0]);
}


__host__ __device__
void Physics::AddElasticForces(
	float F[12], 
	const float eps[9], 
	const float Ainv[16], 
	const float a[4], 
	const float b[4], 
	const float c[4], 
	const float &cxxxx, 
	const float &cxxyy, 
	const float &cxyxy)
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
	float F[12], 
	const float Q[9], 
	const float Ainv[16], 
	const float a[4], 
	const float b[2], 
	const float c[4], 
	const float &alpha)
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
	float a[4], 
	float b[4], 
	float c[4], 
	const float r[12], 
	const float r0[12], 
	const float Ainv[16])
{
	float u[4], v[4] ,w[4];
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




