#ifndef __UPDATAKERNEL_H__
#define __UPDATAKERNEL_H__

#include "sumForce.h"
#include "update_r.h"

//================================================
// -read in all forces on each node and sum them
// -use velocity verilet to step system forward
// -send updated positions to global memory 
//================================================
/*__global__ void updateKernel(	 float *dF*/
/*				,int pitchdF*/
/*				,float *F*/
/*				,int pitchF*/
/*				,int Nnodes */
/*				,int *NodeRank*/
/*				,float *v*/
/*				,int pitchv*/
/*				,float *r*/
/*				,int pitchr	*/
/*				,float *mass*/
/*				,float xclamp1, float xclamp2*/
/*				,float ztable){  // puts sim on table*/
__global__ void updateKernel(DevDataBlock data, float xclamp1, float xclamp2, float ztable)
{	
	int dFshift = data.dFpitch/sizeof(float);
	int Fshift = data.Fpitch/sizeof(float);
	int vshift = data.vpitch/sizeof(float);
	int rshift = data.rpitch/sizeof(float);
	int myNode;
	int myNodeRank;
	float Fnew[3]={0.0};
	float Fold[3];
	float vold[3];
	float vnew[3];
	float localMass;
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
		float xclamps[2] = { xclamp1, xclamp2 };
		update_r(data.r, rshift, vnew, Fnew, myNode, localMass, xclamps, ztable);

	}//tid<Nnodes
}//updateKernel


#endif//__UPDATAKERNEL_H__
