#ifndef __SUMFORCE_H__
#define __SUMFORCE_H__

#include "mainhead.h"

//===========================================
// read in a sum all forces present on node 
// being executed on this thread
//===========================================
__device__ void sumForce( int myNode
						, int myNodeRank
						, float (&Fnew)[3]
						, float (&Fold)[3]
						, float (&vold)[3]
						, float *dF
						, int dFshift
						, float *F
						, int Fshift
						, float *v
						, int vshift){


//float fstretch = 0.01;
//float ry = tex2D(texRef_r0,myNode,1);

/*
if(ry<3.0){
	Fnew[1]=-fstretch;
}
if(ry>197.0){
	Fnew[1]=fstretch;
}
*/

	for( int cord=0;cord<3;cord++ ){

		//sum forces from each tet this node belongs to 
		for(int r=0;r<myNodeRank;r++){Fnew[cord]+=dF[dFshift*(cord+3*r)+myNode];}
		

		//read in old forces
		Fold[cord] = F[Fshift*cord+myNode];

		//set old forces to new forces
		F[Fshift*cord+myNode]=Fnew[cord];

		//read in old velocities
		vold[cord] = v[vshift*cord+myNode];

	}
	


}//sumForce



#endif //__SUMFORCE_H__