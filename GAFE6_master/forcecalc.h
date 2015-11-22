#ifndef __FORCECALC_H__
#define __FORCECALC_H__

#include "mainhead.h"
#include "parameters.h"
#include "UserDefined.h"


//=============================================================
//calculate forces on all 4 nodes in tetrahedra
//=============================================================
__device__ void force_calc(float *Ainv,
						   float *r0,
						   float *r,
						   float *Q,
						   float (&F)[12],
						   int *TetNodeRank,
						   float *pe,
						   int mytet,
						   float myVol){

	float u[4], v[4], w[4];
	float eps[9];
	float a[4] = { 0.0 };
	float b[4] = { 0.0 };
	float c[4] = { 0.0 };
	float localPe = 0.0;
	

	//clacluate displacements from original position and zero out forces
	for(int n = 0; n < 4; n++){
		u[n] = (r[n * 3] - r0[n * 3]);
		v[n] = (r[1 + n * 3] - r0[1 + n * 3]);
		w[n] = (r[2 + n * 3] - r0[2 + n * 3]);
		F[0 + n * 3] = 0.0;
		F[1 + n * 3] = 0.0;
		F[2 + n * 3] = 0.0;
	}//n

	//matrix multipy Ainv and u,v,w to get shape funcitons
	for(int i = 0; i < 4; i++){
		for(int j = 0; j < 4; j++){
			a[i] += Ainv[4 * i + j] * u[j];
			b[i] += Ainv[4 * i + j] * v[j];
			c[i] += Ainv[4 * i + j] * w[j];
		}//j
	}//i

	//noe we calculate epsilon tensor
	eps[3 * 0 + 0] = a[1] + 0.5*(a[1] * a[1] + b[1] * b[1] + c[1] * c[1]);
	eps[3 * 1 + 1] = b[2] + 0.5*(a[2] * a[2] + b[2] * b[2] + c[2] * c[2]);
	eps[3 * 2 + 2] = c[3] + 0.5*(a[3] * a[3] + b[3] * b[3] + c[3] * c[3]);
	eps[3 * 0 + 1] = 0.5*(a[2] + b[1] + a[1] * a[2] + b[1] * b[2] + c[1] * c[2]);
	eps[3 * 1 + 0] = eps[3 * 0 + 1];
	eps[3 * 0 + 2] = 0.5*(a[3] + c[1] + a[1] * a[3] + b[1] * b[3] + c[1] * c[3]);
	eps[3 * 2 + 0] = eps[3 * 0 + 2];
	eps[3 * 1 + 2] = 0.5*(b[3] + c[2] + a[2] * a[3] + b[2] * b[3] + c[2] * c[3]);
	eps[3 * 2 + 1] = eps[3 * 1 + 2];

	//calculate potential energy
	localPe += cxxxx*(eps[3 * 0 + 0] * eps[3 * 0 + 0] + eps[3 * 1 + 1] * eps[3 * 1 + 1] + eps[3 * 2 + 2] * eps[3 * 2 + 2]);
	localPe += 2.0*cxxyy*(eps[3 * 0 + 0] * eps[3 * 1 + 1] + eps[3 * 1 + 1] * eps[3 * 2 + 2] + eps[3 * 0 + 0] * eps[3 * 2 + 2]);
	localPe += 4.0*cxyxy*(eps[3 * 0 + 1] * eps[3 * 0 + 1] + eps[3 * 1 + 2] * eps[3 * 1 + 2] + eps[3 * 2 + 0] * eps[3 * 2 + 0]);
/*
	localPe += -1.0*alph*(eps[3*0+0]*Q[3*0+0]
				+eps[3*1+1]*Q[3*1+1]+eps[3*2+2]*Q[3*2+2]
				+eps[3*0+1]*Q[3*0+1]+eps[3*1+0]*Q[3*1+0]
				+eps[3*1+2]*Q[3*1+2]+eps[3*2+1]*Q[3*2+1]
				+eps[3*0+2]*Q[3*0+2]+eps[3*2+0]*Q[3*2+0]);
*/

	//send potential to global memory
	pe[mytet] = localPe*myVol;

	//now can calculate forces
	for(int n = 0; n < 4; n++){
		//x
		F[0 + n * 3] += -cxxxx*2.0*eps[3 * 0 + 0] * Ainv[4 * 1 + n] * (1.0 + a[1]);
		F[0 + n * 3] += -cxxxx*2.0*eps[3 * 1 + 1] * Ainv[4 * 2 + n] * a[2];
		F[0 + n * 3] += -cxxxx*2.0*eps[3 * 2 + 2] * Ainv[4 * 3 + n] * a[3];
		F[0 + n * 3] += -2.0*cxxyy*eps[3 * 0 + 0] * Ainv[4 * 2 + n] * a[2];
		F[0 + n * 3] += -2.0*cxxyy*eps[3 * 1 + 1] * Ainv[4 * 1 + n] * (1.0 + a[1]);
		F[0 + n * 3] += -2.0*cxxyy*eps[3 * 1 + 1] * Ainv[4 * 3 + n] * a[3];
		F[0 + n * 3] += -2.0*cxxyy*eps[3 * 2 + 2] * Ainv[4 * 2 + n] * a[2];
		F[0 + n * 3] += -2.0*cxxyy*eps[3 * 2 + 2] * Ainv[4 * 1 + n] * (1.0 + a[1]);
		F[0 + n * 3] += -2.0*cxxyy*eps[3 * 0 + 0] * Ainv[4 * 3 + n] * a[3];
		F[0 + n * 3] += -4.0*cxyxy*eps[3 * 0 + 1] * Ainv[4 * 2 + n] * (1.0 + a[1]);
		F[0 + n * 3] += -4.0*cxyxy*eps[3 * 0 + 1] * Ainv[4 * 1 + n] * a[2];
		F[0 + n * 3] += -4.0*cxyxy*eps[3 * 1 + 2] * Ainv[4 * 3 + n] * a[2];
		F[0 + n * 3] += -4.0*cxyxy*eps[3 * 1 + 2] * Ainv[4 * 2 + n] * a[3];
		F[0 + n * 3] += -4.0*cxyxy*eps[3 * 2 + 0] * Ainv[4 * 3 + n] * (1.0 + a[1]);
		F[0 + n * 3] += -4.0*cxyxy*eps[3 * 2 + 0] * Ainv[4 * 1 + n] * a[3];
		
		//y
		F[1 + n * 3] += -cxxxx*2.0*eps[3 * 0 + 0] * Ainv[4 * 1 + n] * b[1];
		F[1 + n * 3] += -cxxxx*2.0*eps[3 * 1 + 1] * Ainv[4 * 2 + n] * (1.0 + b[2]);
		F[1 + n * 3] += -cxxxx*2.0*eps[3 * 2 + 2] * Ainv[4 * 3 + n] * b[3];
		F[1 + n * 3] += -2.0*cxxyy*eps[3 * 0 + 0] * Ainv[4 * 2 + n] * (1.0 + b[2]);
		F[1 + n * 3] += -2.0*cxxyy*eps[3 * 1 + 1] * Ainv[4 * 1 + n] * b[1];
		F[1 + n * 3] += -2.0*cxxyy*eps[3 * 1 + 1] * Ainv[4 * 3 + n] * b[3];
		F[1 + n * 3] += -2.0*cxxyy*eps[3 * 2 + 2] * Ainv[4 * 2 + n] * (1.0 + b[2]);
		F[1 + n * 3] += -2.0*cxxyy*eps[3 * 2 + 2] * Ainv[4 * 1 + n] * b[1];
		F[1 + n * 3] += -2.0*cxxyy*eps[3 * 0 + 0] * Ainv[4 * 3 + n] * b[3];
		F[1 + n * 3] += -4.0*cxyxy*eps[3 * 0 + 1] * Ainv[4 * 1 + n] * (1.0 + b[2]);
		F[1 + n * 3] += -4.0*cxyxy*eps[3 * 0 + 1] * Ainv[4 * 2 + n] * b[1];
		F[1 + n * 3] += -4.0*cxyxy*eps[3 * 1 + 2] * Ainv[4 * 3 + n] * (1.0 + b[2]);
		F[1 + n * 3] += -4.0*cxyxy*eps[3 * 1 + 2] * Ainv[4 * 2 + n] * b[3];
		F[1 + n * 3] += -4.0*cxyxy*eps[3 * 2 + 0] * Ainv[4 * 3 + n] * b[1];
		F[1 + n * 3] += -4.0*cxyxy*eps[3 * 2 + 0] * Ainv[4 * 1 + n] * b[3];

		//z
		F[2 + n * 3] += -cxxxx*2.0*eps[3 * 0 + 0] * Ainv[4 * 1 + n] * c[1];
		F[2 + n * 3] += -cxxxx*2.0*eps[3 * 1 + 1] * Ainv[4 * 2 + n] * c[2];
		F[2 + n * 3] += -cxxxx*2.0*eps[3 * 2 + 2] * Ainv[4 * 3 + n] * (1.0 + c[3]);
		F[2 + n * 3] += -2.0*cxxyy*eps[3 * 0 + 0] * Ainv[4 * 2 + n] * c[2];
		F[2 + n * 3] += -2.0*cxxyy*eps[3 * 1 + 1] * Ainv[4 * 1 + n] * c[1];
		F[2 + n * 3] += -2.0*cxxyy*eps[3 * 1 + 1] * Ainv[4 * 3 + n] * (1.0 + c[3]);
		F[2 + n * 3] += -2.0*cxxyy*eps[3 * 2 + 2] * Ainv[4 * 2 + n] * c[2];
		F[2 + n * 3] += -2.0*cxxyy*eps[3 * 2 + 2] * Ainv[4 * 1 + n] * c[1];
		F[2 + n * 3] += -2.0*cxxyy*eps[3 * 0 + 0] * Ainv[4 * 3 + n] * (1.0 + c[3]);
		F[2 + n * 3] += -4.0*cxyxy*eps[3 * 0 + 1] * Ainv[4 * 2 + n] * c[1];
		F[2 + n * 3] += -4.0*cxyxy*eps[3 * 0 + 1] * Ainv[4 * 1 + n] * c[2];
		F[2 + n * 3] += -4.0*cxyxy*eps[3 * 1 + 2] * Ainv[4 * 2 + n] * (1.0 + c[3]);
		F[2 + n * 3] += -4.0*cxyxy*eps[3 * 1 + 2] * Ainv[4 * 3 + n] * c[2];
		F[2 + n * 3] += -4.0*cxyxy*eps[3 * 2 + 0] * Ainv[4 * 1 + n] * (1.0 + c[3]);
		F[2 + n * 3] += -4.0*cxyxy*eps[3 * 2 + 0] * Ainv[4 * 3 + n] * c[1];

		//force from Q
		F[0 + n * 3] += alph*(Q[0 * 3 + 0] * Ainv[4 * 1 + n] * (1.0 + a[1])
			+ Q[1 * 3 + 1] * Ainv[4 * 2 + n] * a[2]
			+ Q[2 * 3 + 2] * Ainv[4 * 3 + n] * a[3]
			+ Q[0 * 3 + 1] * (Ainv[4 * 2 + n] * (1.0 + a[1]) + Ainv[4 * 1 + n] * a[2])
			+ Q[0 * 3 + 2] * (Ainv[4 * 3 + n] * (1.0 + a[1]) + Ainv[4 * 1 + n] * a[3])
			+ Q[1 * 3 + 2] * (Ainv[4 * 3 + n] * a[2] + Ainv[4 * 2 + n] * a[3]));
	
		F[1 + n * 3] += alph*(Q[0 * 3 + 0] * Ainv[4 * 1 + n] * b[1]
			+ Q[1 * 3 + 1] * Ainv[4 * 2 + n] * (1.0 + b[2])
			+ Q[2 * 3 + 2] * Ainv[4 * 3 + n] * b[3]
			+ Q[0 * 3 + 1] * (Ainv[4 * 1 + n] * (1.0 + b[2]) + Ainv[4 * 2 + n] * b[1])
			+ Q[0 * 3 + 2] * (Ainv[4 * 3 + n] * b[1] + Ainv[4 * 1 + n] * b[3])
			+ Q[1 * 3 + 2] * (Ainv[4 * 3 + n] * (1.0 + b[2]) + Ainv[4 * 2 + n] * b[3]));

		F[2 + n * 3] += alph*(Q[0 * 3 + 0] * Ainv[4 * 1 + n] * c[1]
			+ Q[1 * 3 + 1] * Ainv[4 * 2 + n] * c[2]
			+ Q[2 * 3 + 2] * Ainv[4 * 3 + n] * (1.0 + c[3])
			+ Q[0 * 3 + 1] * (Ainv[4 * 2 + n] * c[1] + Ainv[4 * 1 + n] * c[2])
			+ Q[0 * 3 + 2] * (Ainv[4 * 1 + n] * (1.0 + c[3]) + Ainv[4 * 3 + n] * c[1])
			+ Q[1 * 3 + 2] * (Ainv[4 * 2 + n] * (1.0 + c[3]) + Ainv[4 * 3 + n] * c[2]));
		
	}//n

	//add user calculated force (UserDefined.h)
	userForce(r,Q,F);


}//force_calc


//===========================================================
//calculate drag forces, equal and oposite between neighbors
//----NOT WORKING
//===========================================================
/*
__device__ void calc_drag(float *v,float *r,float (&F)[12]){
	float dx,dy,dz,RR,R,dvx,dvy,dvz,ff,ffx,ffy,ffz;

	

	for(int i=0;i<3;i++){
		for(int j=i+1;j<4;j++){

			dx = r[3*j]-r[3*i];
			dy = r[1+3*j]-r[1+3*i];
			dz = r[2+3*j]-r[2+3*i];

			RR = dx*dx+dy*dy+dz*dz;
			R = sqrt(RR);

			dvx = v[3*j]-v[3*i];
			dvy = v[1+3*j]-v[1+3*i];
			dvz = v[2+3*j]-v[2+3*i];

			ff = -gamma*(dvx*dx+dvy*dy+dvz*dz)/(RR*R);

			ffx = ff*dx;
			ffy = ff*dy;
			ffz = ff*dz;

			//update F
			F[3*i]+=-ffx;
			F[3*j]+=ffx;
			F[1+3*i]+=-ffy;
			F[1+3*j]+=ffy;
			F[2+3*i]+=-ffz;
			F[2+3*j]+=ffz;


		}//j
	}//i

}//calc_drag
*/


#endif//__FORCECALC_H__
