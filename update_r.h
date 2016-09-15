#ifndef __UPDATE_R_H__
#define __UPDATE_R_H__

#include "mainhead.h"
#include "parameters.h"


//=============================================================
// update velocities 
//=============================================================
__device__ void update_v(float (&vnew)[3]
			,float *vold
			,float *Fold
			,float *Fnew
			,float *v
			,int vshift
			,int myNode
			,float mass){
	for (int cord=0;cord<3;cord++){
		vnew[cord] = vold[cord]*damp+ dto2*((Fold[cord]+Fnew[cord])/mass);  //--[ mm / s ]
		v[vshift*cord+myNode]=vnew[cord];
	}//cord

}//update_v



//=============================================================
// calculate change in positions
//=============================================================
__device__ void update_r( float *r
			,int rshift
			,float *vnew
			,float *Fnew
			,int myNode
			,float mass
			,float xclamps[2]
			,float ztable){
//update only if not on edge
const float x = r[rshift*0+myNode];
const float z = r[rshift*2+myNode];
if (x>xclamps[0] && x<xclamps[1] ){  //clamps both ends of LCE
	//update new r's from new v and new F
	for (int cord=0;cord<3;cord++){
		r[rshift*cord+myNode] += dt*vnew[cord]+dt2o2*(Fnew[cord]/mass); //--[ mm ]
	}//i

	if(z<ztable) r[rshift*1+myNode] = ztable; // puts sim on a table
}//if rshift

}//update_r




#endif//__UPDATE_R_H__
