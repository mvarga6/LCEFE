#ifndef __DATASTRUCT_H__
#define __DATASTRUCT_H__

struct DevDataBlock {
	float *dev_A;
	int *dev_TetToNode;
	float *dev_r0;
	float *dev_r;
	float *dev_F;
	float *dev_dF;
	float *dev_v;
	int *dev_nodeRank;
	int *dev_TetNodeRank;
	float *dev_dr;
	float *dev_m;
	float *dev_pe;
	float *dev_TetVol;
	int *dev_ThPhi;
	size_t dev_Apitch;
	size_t dev_TetToNodepitch;
	size_t dev_r0pitch;
	size_t dev_rpitch;
	size_t dev_Fpitch;
	size_t dev_vpitch;
	size_t dev_drpitch;
	size_t dev_dFpitch;


	cudaEvent_t     start, stop;
    float           totalTime;
};

struct HostDataBlock {
	float *host_A;
	int *host_TetToNode;
	float *host_r0;
	float *host_r;
	float *host_F;
	float *host_v;
	int *host_nodeRank;
	int *host_TetNodeRank;
	float *host_dr;
	float *host_m;
	float *host_pe;
	float host_totalVolume;
	float *host_TetVol;
	int *host_ThPhi;

};


#endif //__DATASTRUCT_H__
