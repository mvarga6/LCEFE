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
					,float t)       //time
<<<<<<< HEAD
					/*,float dt*/
=======
					,float t_on 	//total time element has been illuminated in simulation
					,float S_prev	//previous order parameter
					,float L		//illumination amount
>>>>>>> ffdc7935f50f33ffa9d2a4332054562431410cc9
					{

// new calculation of S
// Magnitude of order parameters relaxes back
// to equilibrium with time const tau_S after
// being illuminated by light source.
// S(dt) = S0 * {1 - exp(-dt/tau_S)}

<<<<<<< HEAD
// old calculation of S
float oneThird = 1.0/3.0;
float mythphi = float(myThPhi);
float S=-1.0*t/0.2;
if(S<-1.0){S=-1.0;}
=======
const float oneThird = 1.0/3.0;
const float mythphi = float(myThPhi);
const float S0 = 1.0f; // starting order parameter
const float tau = 0.001f; // trans-cis transition characteristic time
const float t_off = t - t_on;

//calculate S as sigmoid function:
// if {t_on - _t_off = 0} then {S = S0 / 2}
// if {t_on >> t_off}     then {S --> S0}
// if {t_on << t_off}	  then {S --> 0} 
const float S = S0 * sigmoid((t_on - t_off)/tau);
// or maybe this
//const float a = 0.5f, b = 0.5f;
//const float S = sigmoid(a*S_prev + b*L);

//old calculation
//float S=-1.0*t/0.2;
//if(S<-1.0){S=-1.0;}
>>>>>>> ffdc7935f50f33ffa9d2a4332054562431410cc9

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
