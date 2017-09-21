#ifndef __GPUFORCE_H__
#define __GPUFORCE_H__
#include "mainhead.h"
#include "sendForce.h"
#include "read_dev_data.h"
#include "getQ.h"


/*__global__ void force_kernel(	float *A*/
/*								,int pitchA*/
/*								,float *dF*/
/*								,int pitchdF*/
/*								,int *TetNodeRankG*/
/*								,int Ntets*/
/*								,float *v*/
/*								,int pitchv*/
/*								,float *pe*/
/*								,float *TetVol*/
/*								,int *ThPhi*/
/*								,int *S //order parameter*/
/*								,int *L //illumination parameter*/
/*								,int *TetToNode*/
/*								,int pitchTetToNode*/
/*								,float t*/
/*								){*/
__global__ void force_kernel(DevDataBlock data, float t)
{


	int Ashift = data.Apitch/sizeof(float);
	int dFshift = data.dFpitch/sizeof(float);
	int vshift = data.vpitch/sizeof(float);
	int TTNshift = data.TetToNodepitch/sizeof(int);
	float Ainv[16];
	float r[12];
	float r0[12];
	float F[12]={0.0};
	float vlocal[12];
	int node_num[4];
	int TetNodeRank[4];
	float Q[9] = {0.0};
	float myVol;

	//thread ID
	int tid = threadIdx.x + blockIdx.x * blockDim.x;


	if(tid < data.Ntets){ //if thread executed has a tetrahedra


		//========================================
		//read in all the data that will not change 
		//though entire simulation
		//========================================
		myVol = data.TetVol[tid];   //simple enough here

		get_initial_data(Ainv
						,r0
						,node_num
						,Ashift
						,data.A
						,data.v
						,vshift
						,vlocal
						,data.TetNodeRank
						,TetNodeRank
						,data.TetToNode
						,TTNshift
						,data.Ntets);

		//========================================
		//read in data that will be changing
		//IOswitch sets which to read from to
		//allow use of texture memory
		//========================================
		get_variable_data(r, node_num);

		//========================================
		//Calculate illumination on this tetrahedra
		//with rays from nodes to light source
		//========================================
		//illumination(r,node_num,dt_since_illum,light_source)

		//========================================
		//Calcuate Q as a function of Position
		//and time for this tetrahedra
		//
		//	-- NEW --
		// Send S and L to Q calculation and update
		// S for next calculation.
		//========================================
		getQ(data.ThPhi[tid], Q, t, data.S[tid], data.L[tid]); // just for debugging

		//========================================
		//calculate the force on each node due
		//to interactions in this tetrahedra
		//========================================
		force_calc(Ainv, r0, r, Q, F, TetNodeRank, data.pe, tid, myVol);

		//========================================
		//Send each force calculated to global 
		//memroy so force can be summed in 
		//update kernal
		//========================================
		sendForce(data.dF, dFshift, F, node_num, TetNodeRank, myVol);


	}//end if tid<Ntets

}//end force kernel


#endif //__gpuForce_H__
