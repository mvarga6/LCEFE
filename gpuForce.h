#ifndef __GPUFORCE_H__
#define __GPUFORCE_H__
#include "mainhead.h"
#include "sendForce.h"
#include "read_dev_data.h"
#include "getQ.h"
#include "device_helpers.h"

/*__global__ void force_kernel(	real *A*/
/*								,int pitchA*/
/*								,real *dF*/
/*								,int pitchdF*/
/*								,int *TetNodeRankG*/
/*								,int Ntets*/
/*								,real *v*/
/*								,int pitchv*/
/*								,real *pe*/
/*								,real *TetVol*/
/*								,int *ThPhi*/
/*								,int *S //order parameter*/
/*								,int *L //illumination parameter*/
/*								,int *TetToNode*/
/*								,int pitchTetToNode*/
/*								,real t*/
/*								){*/
__global__ void force_kernel(DevDataBlock data, real t)
{


	int Ashift = data.Apitch/sizeof(real);
	int dFshift = data.dFpitch/sizeof(real);
	int vshift = data.vpitch/sizeof(real);
	int TTNshift = data.TetToNodepitch/sizeof(int);
	real Ainv[16];
	real r[12];
	real r0[12];
	real F[12]={0.0};
	real vlocal[12];
	int NodeNum[4];
	int TetNodeRank[4];
	real Q[9] = {0.0};
	real myVol;

	//thread ID
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	//if thread executed has a tetrahedra
	if(tid < data.Ntets)
	{ 


		//========================================
		//read in all the data that will not change 
		//though entire simulation
		//========================================
		myVol = data.TetVol[tid];   //simple enough here

/*		get_initial_data(Ainv*/
/*						,r0*/
/*						,NodeNum*/
/*						,Ashift*/
/*						,data.A*/
/*						,data.v*/
/*						,vshift*/
/*						,vlocal*/
/*						,data.TetNodeRank*/
/*						,TetNodeRank*/
/*						,data.TetToNode*/
/*						,TTNshift*/
/*						,data.Ntets);*/

		//========================================
		//read in data that will be changing
		//IOswitch sets which to read from to
		//allow use of texture memory
		//========================================
		//get_variable_data(r, NodeNum);


		//========================================
		//Read all the data needed for force calc
		//========================================
		DeviceHelpers::ReadGlobalToLocal(
			NodeNum, TetNodeRank,
			Ainv, r0, r, vlocal,
			data.A, Ashift,
			data.v, vshift,
			data.TetNodeRank,
			data.TetToNode, TTNshift,
			data.Ntets
		);

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
		//sendForce(data.dF, dFshift, F, NodeNum, TetNodeRank, myVol);
		DeviceHelpers::SendForce(data.dF, dFshift, F, NodeNum, TetNodeRank, myVol, tid);

#ifdef __DEBUG_FORCE__
		
		// debugging info
		if (tid == __DEBUG_FORCE__)
		{
			printf("\n -- force_kernel --")
			printf("\n\tTime = %f", t);
			printf("\n\tMyVol = %f", myVol);
			printf("\n\n\tF[] = ");
			printf("\n\t1: %.3f %.3f %.3f", F[0], F[1], F[2]);
			printf("\n\t2: %.3f %.3f %.3f", F[3], F[4], F[5]);
			printf("\n\t3: %.3f %.3f %.3f", F[6], F[7], F[8]);
			printf("\n\t4: %.3f %.3f %.3f", F[9], F[10], F[11]);
			printf("\n\n\tQ[] = ");
			printf("\n\t1: %.3f %.3f %.3f", Q[0], Q[1], Q[2]);
			printf("\n\t2: %.3f %.3f %.3f", Q[3], Q[4], Q[5]);
			printf("\n\t3: %.3f %.3f %.3f", Q[6], Q[7], Q[8]);
		}
#endif

	}//end if tid<Ntets

}//end force kernel


#endif //__gpuForce_H__
