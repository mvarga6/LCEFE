#ifndef __UPDATE_R_H__
#define __UPDATE_R_H__

#include "mainhead.h"
#include "parameters.h"
#include "kernel_constants.h"
#include "defines.h"

//=============================================================
// update velocities 
//=============================================================
__device__ void update_v(real (&vnew)[3]
			,real *vold
			,real *Fold
			,real *Fnew
			,real *v
			,int vshift
			,int myNode
			,real mass)
{

	const real damp = Parameters.Damp;
	const real dto2 = Parameters.Dt / 2.0f;
	
	for (int c = 0; c < 3; c++)
	{
		vnew[c] = vold[c] * damp + dto2 * ((Fold[c] + Fnew[c]) / mass);  //--[ mm / s ]
		v[vshift * c + myNode] = vnew[c];
	}
	
	#ifdef __DEBUG_UPDATE_V__
	if (myNode == __DEBUG_UPDATE_V__)
	{
		printf("\n\n -- update_v --");
		printf("\n\tdto2 = %f", dto2);
		printf("\n\tvold = { %.4f, %.4f, %.4f }", vold[0], vold[1], vold[2]);
		printf("\n\tvnew = { %.4f, %.4f, %.4f }", vnew[0], vnew[1], vnew[2]);
	}
#endif

}//update_v



//=============================================================
// calculate change in positions
//=============================================================
__device__ void update_r( real *r
			,int rshift
			,real *v
			,real *F
			,int myNode
			,real mass)
{
	const real dt = Parameters.Dt;
	const real dt2o2 = (dt*dt) / 2.0f;
	real dr[3] = { 0.0f, 0.0f, 0.0f };
	
	//update new r's from new v and new F
	for (int c = 0; c < 3; c++)
	{
		dr[c] = dt * v[c] + dt2o2 * (F[c] / mass); //--[ mm ]
		r[rshift * c + myNode] += dr[c];
	}
	
#ifdef __DEBUG_UPDATE_R__
	if (myNode == __DEBUG_UPDATE_R__)
	{
		printf("\n\n -- update_r --");
		printf("\n\tF = { %.5f, %.5f, %.5f}", F[0], F[1], F[2]);
		printf("\n\tdr = { %.2f, %.2f, %.2f }", dr[0], dr[1], dr[2]);
		printf("\n\tr = { %.2f, %.2f, %.2f }", r[myNode], r[rshift + myNode], r[rshift*2 + myNode]);
	}
#endif

}//update_r




#endif//__UPDATE_R_H__
