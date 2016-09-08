#ifndef __GETQ_H__
#define __GETQ_H__

#include "mainhead.h"
#include "parameters.h"
#include <math.h>

__device__ float sigmoid(const float &x){
	return 1.0f/(1.0f + exp(-x));
}

//==============================================
//Calculate what Q should be for this tetrahedra 
//given its position (initial) and time
//==============================================
__device__ void getQ(int myThPhi    //theta and Phi
					,float (&Q)[9]  //array to store Q in
					,float t       //time
					//,float t_on 	//total time element has been illuminated in simulation
					,int &S_in	//previous order parameter to calc/set new
					,int L		//illumination amount
					){


const float oneThird = 1.0/3.0;
const float mythphi = float(myThPhi);
//const float tau = 0.001f; // trans-cis transition characteristic time
//const float t_off = t - t_on;

//calculate S as sigmoid function:
// if {t_on - _t_off = 0} then {S = S0 / 2}
// if {t_on >> t_off}     then {S --> S0}
// if {t_on << t_off}	  then {S --> 0} 
// maybe this
//const float S = S0 * sigmoid((t_on - t_off)/tau);
// or maybe this
//const float fS_in = float(S_in) / SRES; //map to float range
//const float fL = float(L);
//const float a = 0.5f, b = 0.5f;
//const float S = fS_in;
//const float S = S0 * sigmoid(a*fS_in + b*fL);

//old calculation
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

// set S with new updated value
S_in = int(S*SRES);

}//end get Q


#endif //__GETQ_H__
