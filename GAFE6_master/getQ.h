#ifndef __GETQ_H__
#define __GETQ_H__

#include "mainhead.h"
#include "parameters.h"
#include <math.h>

#define loopij for(int i = 0; i < 3; i++) for(int j = 0; j < 3; j++)
__constant__ __device__ const float oneThird = 1.0f / 3.0f;
__constant__ __device__ const float delo3[3][3] = 
{ 
	{ 1.0f / 3.0f, 0, 0 }, 
	{ 0, 1.0f / 3.0f, 0 }, 
	{ 0, 0, 1.0f / 3.0f } 
};

//==============================================
//Calculate what Q should be for this tetrahedra 
//given its position (initial) and time
//==============================================
__device__ void getQ(int myThPhi    //theta and Phi
					,float (&Q)[9]  //array to store Q in
					,float t){      //time

	/*__shared__ float oneThird;
	oneThird = 1.0f / 3.0f;
	__shared__ float S;*/
	//float S = -1.0*t / 0.2;
	//if (S < -1.0) S = -1.0;
	float S = t * 2.0f;
	if (S > 0.7f) S = 0.7f;

	//convert ThPhi into theta and phi
	const float mythphi = float(myThPhi);
	float nTh = floor(mythphi / 10000.0f);
	float nPhi = mythphi - nTh*10000.0f;
	float theta = nTh*PI / 1000.0f;
	float phi = nPhi*PI / 500.0f;

	//calculate nx ny and nz from theta and phi
	float n[3], nx, ny, nz;
	n[0] = nx = cosf(phi)*sinf(theta);
	n[1] = ny = sinf(phi)*sinf(theta);
	n[2] = nz = cosf(theta);

	//calculate Q from nx,ny,nz and S
	//loopij Q[i * 3 + j] = S*(n[i] * n[j] - delo3[i][j]);

	Q[0 * 3 + 0] = S*(nx*nx - oneThird);
	Q[1 * 3 + 1] = S*(ny*ny - oneThird);
	Q[2 * 3 + 2] = S*(nz*nz - oneThird);
	Q[0 * 3 + 1] = S*nx*ny;
	Q[0 * 3 + 2] = S*nx*nz;
	Q[1 * 3 + 0] = Q[0 * 3 + 1];
	Q[1 * 3 + 2] = S*ny*nz;
	Q[2 * 3 + 1] = Q[1 * 3 + 2];
	Q[2 * 3 + 0] = Q[0 * 3 + 2];
}//end get Q


#endif //__GETQ_H__
