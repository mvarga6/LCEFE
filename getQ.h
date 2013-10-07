#ifndef __GETQ_H__
#define __GETQ_H__

#include "mainhead.h"
#include "parameters.h"
#include <math.h>




//==============================================
//Calculate what Q should be for this tetrahedra 
//given its position (initial) and time
//==============================================
__device__ void getQ(int myThPhi    //theta and Phi
					,float (&Q)[9]  //array to store Q in
					,float t)       //time
					{


float oneThird = 1.0/3.0;
float mythphi = float(myThPhi);
float S=-1.0*t/0.2;
if(S<-1.0){S=-1.0;}

//convert ThPhi into theta and phi
float nTh,nPhi,theta,phi;
nTh = floor(mythphi/10000.0);
nPhi = mythphi-nTh*10000.0;

theta = nTh*PI/1000.0;
phi = nPhi*PI/500.0;

//calculate nx ny and nz from theta and phi
float nx,ny,nz;
nx = sin(theta)*cos(phi);
ny = sin(theta)*sin(phi);
nz = cos(theta);

//calculate Q from nx,ny,nz and S
Q[0*3+0]=S*(nx*nx-oneThird);
Q[1*3+1]=S*(ny*ny-oneThird);
Q[2*3+2]=S*(nz*nz-oneThird);
Q[0*3+1]=S*nx*ny;
Q[0*3+2]=S*nx*nz;
Q[1*3+0]=Q[0*3+1];
Q[1*3+2]=S*ny*nz;
Q[2*3+1]=Q[1*3+2];
Q[2*3+0]=Q[0*3+2];

}//end get Q


#endif //__GETQ_H__
