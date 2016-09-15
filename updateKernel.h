#ifndef __UPDATAKERNEL_H__
#define __UPDATAKERNEL_H__

#include "sumForce.h"
#include "update_r.h"

//================================================
// -read in all forces on each node and sum them
// -use velocity verilet to step system forward
// -send updated positions to global memory 
//================================================
__global__ void updateKernel(	 float *dF
				,int pitchdF
				,float *F
				,int pitchF
				,int Nnodes 
				,int *NodeRank
				,float *v
				,int pitchv
				,float *r
				,int pitchr	
				,float *mass
				,float xclamp1, float xclamp2
				,float ztable){  // puts sim on table
	
	int dFshift = pitchdF/sizeof(float);
	int Fshift = pitchF/sizeof(float);
	int vshift = pitchv/sizeof(float);
	int rshift = pitchr/sizeof(float);
	int myNode;
	int myNodeRank;
	float Fnew[3]={0.0};
	float Fold[3];
	float vold[3];
	float vnew[3];
	float localMass;
	//thread ID
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid<Nnodes){  //if a node is here
		myNode=tid;
		myNodeRank = NodeRank[myNode];
		localMass = mass[myNode];

		//get new and old forces + old velocities
		sumForce(myNode,myNodeRank,Fnew,Fold,vold,dF,dFshift,F,Fshift,v,vshift);

		//calculate and store new velocites
		update_v(vnew,vold,Fold,Fnew,v,vshift,myNode,localMass);

		//calculate and store new positions
		float xclamps[2] = { xclamp1, xclamp2 };
		update_r(r,rshift,vnew,Fnew,myNode,localMass,xclamps,ztable);

	}//tid<Nnodes
}//updateKernel


#endif//__UPDATAKERNEL_H__
