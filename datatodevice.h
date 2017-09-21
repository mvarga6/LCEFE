#ifndef __DATATODEVICE_H__
#define __DATATODEVICE_H__

#include "mainhead.h"

void data_to_device(DevDataBlock *dev, HostDataBlock *host){


	//need to pitch 1D memory correctly to send to device
	int Nnodes = host->Nnodes;
	int Ntets = host->Ntets;
	size_t height16 = 16;
	size_t height4 = 4;
	size_t height3 = 3;
	size_t heightMR = MaxNodeRank*3;
	size_t widthNODE = Nnodes;
	size_t widthTETS = Ntets;
	
	dev->Nnodes = Nnodes;
	dev->Ntets = Ntets;

	//set offset to be 0
	size_t offset = 0;

	//used pitch linear memory on device for fast access
	//allocate memory on device for pitched linear memory

	HANDLE_ERROR( cudaMallocPitch( (void**) &dev->A 
									, &dev->Apitch 
									, widthTETS*sizeof(float) 
									, height16 ) );
	HANDLE_ERROR( cudaMallocPitch( (void**) &dev->dF 
									, &dev->dFpitch 
									, widthNODE*sizeof(float) 
									, heightMR ) );
	HANDLE_ERROR( cudaMallocPitch( (void**) &dev->TetToNode 
									, &dev->TetToNodepitch 
									, widthTETS*sizeof(float) 
									, height4 ) );
	HANDLE_ERROR( cudaMallocPitch( (void**) &dev->r0 
									, &dev->r0pitch  
									, widthNODE*sizeof(float) 
									, height3 ) );
	HANDLE_ERROR( cudaMallocPitch( (void**) &dev->r 
									, &dev->rpitch 
									, widthNODE*sizeof(float) 
									, height3 ) );
	HANDLE_ERROR( cudaMallocPitch( (void**) &dev->v 
									, &dev->vpitch 
									, widthNODE*sizeof(float) 
									, height3 ) );
	HANDLE_ERROR( cudaMallocPitch( (void**) &dev->F, &dev->Fpitch
									, widthNODE*sizeof(float) 
									, height3 ) );

	HANDLE_ERROR( cudaMalloc( (void**) &dev->TetNodeRank, Ntets*4*sizeof(int) ) );
	HANDLE_ERROR( cudaMalloc( (void**) &dev->nodeRank, Nnodes*sizeof(int) ) );
	HANDLE_ERROR( cudaMalloc( (void**) &dev->m, Nnodes*sizeof(float) ) );
	HANDLE_ERROR( cudaMalloc( (void**) &dev->pe, Ntets*sizeof(float) ) );
	HANDLE_ERROR( cudaMalloc( (void**) &dev->TetVol, Ntets*sizeof(float) ) );
	HANDLE_ERROR( cudaMalloc( (void**) &dev->ThPhi, Ntets*sizeof(int) ) );
	HANDLE_ERROR( cudaMalloc( (void**) &dev->S, Ntets*sizeof(int) ) );
	HANDLE_ERROR( cudaMalloc( (void**) &dev->L, Ntets*sizeof(int) ) );

	//copy data to device
	//This will copy each 1D array as if it 
	//is a pitched linear array which can be accessed like
	//a 2-D array

	HANDLE_ERROR( cudaMemcpy(dev->TetNodeRank
								,host->TetNodeRank
								,Ntets*4*sizeof(int)
								,cudaMemcpyHostToDevice) );
	HANDLE_ERROR( cudaMemcpy(dev->ThPhi
								,host->ThPhi
								,Ntets*sizeof(int)
								,cudaMemcpyHostToDevice) );
	HANDLE_ERROR( cudaMemcpy(dev->nodeRank
								,host->nodeRank
								,Nnodes*sizeof(int)
								,cudaMemcpyHostToDevice) );
	HANDLE_ERROR( cudaMemcpy(dev->m
								,host->m
								,Nnodes*sizeof(float)
								,cudaMemcpyHostToDevice) );
	HANDLE_ERROR( cudaMemcpy(dev->TetVol
								,host->TetVol
								,Ntets*sizeof(float)
								,cudaMemcpyHostToDevice) );

	HANDLE_ERROR( cudaMemcpy(dev->S
								,host->S
								,Ntets*sizeof(int)
								,cudaMemcpyHostToDevice) );
	HANDLE_ERROR( cudaMemset( dev->L, 0, Ntets*sizeof(int) ) );
	HANDLE_ERROR( cudaMemcpy2D( dev->A
								, dev->Apitch
								, host->A
								, widthTETS*sizeof(float)
								, widthTETS*sizeof(float)
                                , height16
								, cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy2D( dev->TetToNode
								, dev->TetToNodepitch
								, host->TetToNode
								, widthTETS*sizeof(int)
								, widthTETS*sizeof(int)
								, height4
								, cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy2D( dev->r0
								, dev->r0pitch
								, host->r0
								, widthNODE*sizeof(float)
								, widthNODE*sizeof(float)
								, height3
								, cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy2D( dev->r
								, dev->rpitch
								, host->r
								, widthNODE*sizeof(float)
								, widthNODE*sizeof(float)
								, height3
								, cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy2D( dev->v
								, dev->vpitch
								, host->v
                                , widthNODE*sizeof(float)
								, widthNODE*sizeof(float)
                                , height3
								, cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy2D( dev->F
								, dev->Fpitch
								, host->F
                                , widthNODE*sizeof(float)
								, widthNODE*sizeof(float)
                                , height3
								, cudaMemcpyHostToDevice ) );

	//================================================
	//bind linear pitched memory to 2D texture
	//================================================


	HANDLE_ERROR( cudaBindTexture2D( &offset 
									,texRef_r0
									, dev->r0
									, texRef_r0.channelDesc
									, widthNODE
									, height3
									, dev->r0pitch) );
	texRef_r0.normalized = false;
	//texRef_r0.filterMode = cudaFilterModeLinear;
	HANDLE_ERROR( cudaBindTexture2D( &offset 
									, texRef_r
									, dev->r
									, texRef_r.channelDesc
									, widthNODE
									, height3
									, dev->rpitch) );
	texRef_r.normalized = false;
	//texRef_rA.filterMode = cudaFilterModeLinear;
	printf("\ndata sent to device\n");

}


#endif//__DATATODEVICE_H__
