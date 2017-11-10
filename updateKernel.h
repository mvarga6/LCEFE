#ifndef __UPDATAKERNEL_H__
#define __UPDATAKERNEL_H__

#include "sumForce.h"
#include "update_r.h"

//================================================
// -read in all forces on each node and sum them
// -use velocity verilet to step system forward
// -send updated positions to global memory 
//================================================
__global__ void updateKernel(DevDataBlock data)
{	
	int dFshift = data.dFpitch/sizeof(real);
	int Fshift = data.Fpitch/sizeof(real);
	int vshift = data.vpitch/sizeof(real);
	int rshift = data.rpitch/sizeof(real);
	int myNode;
	int myNodeRank;
	real Fnew[3]={0.0};
	real Fold[3];
	real vold[3];
	real vnew[3];
	real localMass;
	//thread ID
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid < data.Nnodes) //if a node is here
	{  
		myNode = tid;
		myNodeRank = data.nodeRank[myNode];
		localMass = data.m[myNode];

		//get new and old forces + old velocities
		sumForce(myNode, myNodeRank, Fnew, Fold, vold, data.dF, dFshift, data.F, Fshift, data.v, vshift);

		//calculate and store new velocites
		update_v(vnew, vold, Fold, Fnew, data.v, vshift, myNode, localMass);

		//calculate and store new positions
		update_r(data.r, rshift, vnew, Fnew, myNode, localMass);

	}//tid<Nnodes
}//updateKernel


#endif//__UPDATAKERNEL_H__
