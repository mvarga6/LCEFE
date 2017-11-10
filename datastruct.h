#ifndef __DATASTRUCT_H__
#define __DATASTRUCT_H__

#include "cuda.h"
#include "cuda_runtime.h"
#include "stdio.h"
#include "defines.h"

struct DevDataBlock 
{
	int Ntets, Nnodes;
	real *A;
	int *TetToNode;
	real *r0;
	real *r;
	real *F;
	real *dF;
	real *v;
	int *nodeRank;
	int *TetNodeRank;
	real *dr;
	real *m;
	real *pe;
	real *TetVol;
	int *ThPhi;
	int *S;
	int *L;
	size_t TetToNodepitch;
	size_t Apitch;
	size_t r0pitch;
	size_t rpitch;
	size_t Fpitch;
	size_t vpitch;
	size_t drpitch;
	size_t dFpitch;


	//cudaEvent_t     start, stop;
    //	real           totalTime;
};

struct HostDataBlock 
{
	int Ntets, Nnodes;
	real *A;
	int *TetToNode;
	real *r0;
	real *r;
	real *F;
	real *v;
	int *nodeRank;
	int *TetNodeRank;
	real *dr;
	real *m;
	real *pe;
	real totalVolume;
	real *TetVol;
	int *ThPhi;
	int *S;

	real min[3], max[3];
};


#endif //__DATASTRUCT_H__
