#ifndef __GPUFORCE_H__
#define __GPUFORCE_H__
#include "mainhead.h"
#include "sendForce.h"
#include "read_dev_data.h"
#include "getQ.h"


__global__ void force_kernel(	float *A
								,int pitchA
								,float *dF
								,int pitchdF
								,int *TetNodeRankG
								,int Ntets
								,float *v
								,int pitchv
								,float *pe
								,float *TetVol
								,int *ThPhi
								,int *TetToNode
								,int pitchTetToNode
								,float t){


	int Ashift = pitchA/sizeof(float);
	int dFshift = pitchdF/sizeof(float);
	int vshift = pitchv/sizeof(float);
	int TTNshift = pitchTetToNode/sizeof(int);
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


	if(tid<Ntets){ //if thread executed has a tetrahedra


		//========================================
		//read in all the data that will not change 
		//though entire simulation
		//========================================
		myVol = TetVol[tid];   //simple enough here

		get_initial_data(Ainv
						,r0
						,node_num
						,Ashift
						,A
						,v
						,vshift
						,vlocal
						,TetNodeRankG
						,TetNodeRank
						,TetToNode
						,TTNshift
						,Ntets);

		//========================================
		//read in data that will be changing
		//IOswitch sets which to read from to
		//allow use of texture memory
		//========================================
		get_variable_data(r,node_num);



		//========================================
		//Calcuate Q as a function of Position
		//and time for this tetrahedra
		//========================================
		getQ(ThPhi[tid],Q,t);

		
		//========================================
		//calculate the force on each node due
		//to interactions in this tetrahedra
		//========================================
		force_calc(Ainv,r0,r,Q,F,TetNodeRank,pe,tid,myVol);


		//========================================
		//Send each force calculated to global 
		//memroy so force can be summed in 
		//update kernal
		//========================================
		sendForce(dF,dFshift,F,node_num,TetNodeRank,myVol);


	}//end if tid<Ntets

}//end force kernel


#endif //__gpuForce_H__