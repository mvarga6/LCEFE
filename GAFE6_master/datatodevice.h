#ifndef __DATATODEVICE_H__
#define __DATATODEVICE_H__

#include "mainhead.h"

void data_to_device(DevDataBlock *dev_dat, HostDataBlock *host_dat,int Ntets,int Nnodes){


	//need to pitch 1D memory correctly to send to device
	size_t height16 = 16;
	size_t height4 = 4;
	size_t height3 = 3;
	size_t heightMR = MaxNodeRank*3;
	size_t widthNODE = Nnodes;
	size_t widthTETS = Ntets;

	//set offset to be 0
	size_t offset = 0;



	//used pitch linear memory on device for fast access
	//allocate memory on device for pitched linear memory

	HANDLE_ERROR( cudaMallocPitch( (void**) &dev_dat->dev_A 
									, &dev_dat->dev_Apitch 
									, widthTETS*sizeof(float) 
									, height16 ) );

	HANDLE_ERROR( cudaMallocPitch( (void**) &dev_dat->dev_dF 
									, &dev_dat->dev_dFpitch 
									, widthNODE*sizeof(float) 
									, heightMR ) );

	HANDLE_ERROR( cudaMallocPitch( (void**) &dev_dat->dev_TetToNode 
									, &dev_dat->dev_TetToNodepitch 
									, widthTETS*sizeof(float) 
									, height4 ) );
		

	HANDLE_ERROR( cudaMallocPitch( (void**) &dev_dat->dev_r0 
									, &dev_dat->dev_r0pitch  
									, widthNODE*sizeof(float) 
									, height3 ) );

	HANDLE_ERROR( cudaMallocPitch( (void**) &dev_dat->dev_r 
									, &dev_dat->dev_rpitch 
									, widthNODE*sizeof(float) 
									, height3 ) );


	HANDLE_ERROR( cudaMallocPitch( (void**) &dev_dat->dev_v 
									, &dev_dat->dev_vpitch 
									, widthNODE*sizeof(float) 
									, height3 ) );
	
	HANDLE_ERROR( cudaMallocPitch( (void**) &dev_dat->dev_F 
									, &dev_dat->dev_Fpitch
									, widthNODE*sizeof(float) 
									, height3 ) );

	HANDLE_ERROR( cudaMalloc( (void**) &dev_dat->dev_TetNodeRank
									,Ntets*4*sizeof(int) ) );

	HANDLE_ERROR( cudaMalloc( (void**) &dev_dat->dev_nodeRank
									,Nnodes*sizeof(int) ) );


	HANDLE_ERROR( cudaMalloc( (void**) &dev_dat->dev_m
									,Nnodes*sizeof(float) ) );

	HANDLE_ERROR( cudaMalloc( (void**) &dev_dat->dev_pe
									,Ntets*sizeof(float) ) );
	
	HANDLE_ERROR( cudaMalloc( (void**) &dev_dat->dev_TetVol
									,Ntets*sizeof(float) ) );

	HANDLE_ERROR( cudaMalloc( (void**) &dev_dat->dev_ThPhi
									,Ntets*sizeof(int) ) );

	
	//copy data to device
	//This will copy each 1D array as if it 
	//is a pitched linear array which can be accessed like
	//a 2-D array

	HANDLE_ERROR( cudaMemcpy(dev_dat->dev_TetNodeRank
								,host_dat->host_TetNodeRank
								,Ntets*4*sizeof(int)
								,cudaMemcpyHostToDevice) );

	HANDLE_ERROR( cudaMemcpy(dev_dat->dev_ThPhi
								,host_dat->host_ThPhi
								,Ntets*sizeof(int)
								,cudaMemcpyHostToDevice) );

	HANDLE_ERROR( cudaMemcpy(dev_dat->dev_nodeRank
								,host_dat->host_nodeRank
								,Nnodes*sizeof(int)
								,cudaMemcpyHostToDevice) );

	HANDLE_ERROR( cudaMemcpy(dev_dat->dev_m
								,host_dat->host_m
								,Nnodes*sizeof(float)
								,cudaMemcpyHostToDevice) );
	
	HANDLE_ERROR( cudaMemcpy(dev_dat->dev_TetVol
								,host_dat->host_TetVol
								,Ntets*sizeof(float)
								,cudaMemcpyHostToDevice) );

	HANDLE_ERROR( cudaMemcpy2D( dev_dat->dev_A
								, dev_dat->dev_Apitch
								, host_dat->host_A
								, widthTETS*sizeof(float)
								, widthTETS*sizeof(float)
                                , height16
								, cudaMemcpyHostToDevice ) );

	HANDLE_ERROR( cudaMemcpy2D( dev_dat->dev_TetToNode
								, dev_dat->dev_TetToNodepitch
								, host_dat->host_TetToNode
								, widthTETS*sizeof(int)
								, widthTETS*sizeof(int)
								, height4
								, cudaMemcpyHostToDevice ) );

	HANDLE_ERROR( cudaMemcpy2D( dev_dat->dev_r0
								, dev_dat->dev_r0pitch
								, host_dat->host_r0
								, widthNODE*sizeof(float)
								, widthNODE*sizeof(float)
								, height3
								, cudaMemcpyHostToDevice ) );

	HANDLE_ERROR( cudaMemcpy2D( dev_dat->dev_r
								, dev_dat->dev_rpitch
								, host_dat->host_r
								, widthNODE*sizeof(float)
								, widthNODE*sizeof(float)
								, height3
								, cudaMemcpyHostToDevice ) );


	HANDLE_ERROR( cudaMemcpy2D( dev_dat->dev_v
								, dev_dat->dev_vpitch
								, host_dat->host_v
                                , widthNODE*sizeof(float)
								, widthNODE*sizeof(float)
                                , height3
								, cudaMemcpyHostToDevice ) );


	
	HANDLE_ERROR( cudaMemcpy2D( dev_dat->dev_F
								, dev_dat->dev_Fpitch
								, host_dat->host_F
                                , widthNODE*sizeof(float)
								, widthNODE*sizeof(float)
                                , height3
								, cudaMemcpyHostToDevice ) );



	//================================================
	//bind linear pitched memory to 2D texture
	//================================================


	HANDLE_ERROR( cudaBindTexture2D( &offset 
									,texRef_r0
									, dev_dat->dev_r0
									, texRef_r0.channelDesc
									, widthNODE
									, height3
									, dev_dat->dev_r0pitch) );
	texRef_r0.normalized = false;
	//texRef_r0.filterMode = cudaFilterModeLinear;


	HANDLE_ERROR( cudaBindTexture2D( &offset 
									, texRef_r
									, dev_dat->dev_r
									, texRef_r.channelDesc
									, widthNODE
									, height3
									, dev_dat->dev_rpitch) );
	texRef_r.normalized = false;
	//texRef_rA.filterMode = cudaFilterModeLinear;


	printf("\ndata sent to device\n");

}


#endif//__DATATODEVICE_H__