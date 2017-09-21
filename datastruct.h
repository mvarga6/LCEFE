#ifndef __DATASTRUCT_H__
#define __DATASTRUCT_H__

struct DevDataBlock {
	int Ntets, Nnodes;
	float *A;
	int *TetToNode;
	float *r0;
	float *r;
	float *F;
	float *dF;
	float *v;
	int *nodeRank;
	int *TetNodeRank;
	float *dr;
	float *m;
	float *pe;
	float *TetVol;
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


	cudaEvent_t     start, stop;
    	float           totalTime;
};

struct HostDataBlock {
	int Ntets, Nnodes;
	float *A;
	int *TetToNode;
	float *r0;
	float *r;
	float *F;
	float *v;
	int *nodeRank;
	int *TetNodeRank;
	float *dr;
	float *m;
	float *pe;
	float totalVolume;
	float *TetVol;
	int *ThPhi;
	int *S;

	float min[3], max[3];
};


#endif //__DATASTRUCT_H__
