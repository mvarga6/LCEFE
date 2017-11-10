#ifndef __DATATODEVICE_H__
#define __DATATODEVICE_H__

#include "mainhead.h"
#include "simulation_parameters.h"
#include "kernel_constants.h"

void data_to_device(DevDataBlock *dev, HostDataBlock *host, SimulationParameters *params, DataManager *manager){


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
	//size_t offset = 0;

	//used pitch linear memory on device for fast access
	//allocate memory on device for pitched linear memory

	HANDLE_ERROR( cudaMallocPitch( (void**) &dev->A 
									, &dev->Apitch 
									, widthTETS*sizeof(real) 
									, height16 ) );
	HANDLE_ERROR( cudaMallocPitch( (void**) &dev->dF 
									, &dev->dFpitch 
									, widthNODE*sizeof(real) 
									, heightMR ) );
	HANDLE_ERROR( cudaMallocPitch( (void**) &dev->TetToNode 
									, &dev->TetToNodepitch 
									, widthTETS*sizeof(real) 
									, height4 ) );
	HANDLE_ERROR( cudaMallocPitch( (void**) &dev->r0 
									, &dev->r0pitch  
									, widthNODE*sizeof(real) 
									, height3 ) );
	HANDLE_ERROR( cudaMallocPitch( (void**) &dev->r 
									, &dev->rpitch 
									, widthNODE*sizeof(real) 
									, height3 ) );
	HANDLE_ERROR( cudaMallocPitch( (void**) &dev->v 
									, &dev->vpitch 
									, widthNODE*sizeof(real) 
									, height3 ) );
	HANDLE_ERROR( cudaMallocPitch( (void**) &dev->F, &dev->Fpitch
									, widthNODE*sizeof(real) 
									, height3 ) );

	HANDLE_ERROR( cudaMalloc( (void**) &dev->TetNodeRank, Ntets*4*sizeof(int) ) );
	HANDLE_ERROR( cudaMalloc( (void**) &dev->nodeRank, Nnodes*sizeof(int) ) );
	HANDLE_ERROR( cudaMalloc( (void**) &dev->m, Nnodes*sizeof(real) ) );
	HANDLE_ERROR( cudaMalloc( (void**) &dev->pe, Ntets*sizeof(real) ) );
	HANDLE_ERROR( cudaMalloc( (void**) &dev->TetVol, Ntets*sizeof(real) ) );
	HANDLE_ERROR( cudaMalloc( (void**) &dev->ThPhi, Ntets*sizeof(int) ) );
	HANDLE_ERROR( cudaMalloc( (void**) &dev->S, Ntets*sizeof(int) ) );
	HANDLE_ERROR( cudaMalloc( (void**) &dev->L, Ntets*sizeof(int) ) );
	
	
	DataProcedure *pushAll = new PushAllToGpu();
	manager->Execute(pushAll);

	//copy data to device
	//This will copy each 1D array as if it 
	//is a pitched linear array which can be accessed like
	//a 2-D array

/*	HANDLE_ERROR( cudaMemcpy(dev->TetNodeRank*/
/*								,host->TetNodeRank*/
/*								,Ntets*4*sizeof(int)*/
/*								,cudaMemcpyHostToDevice) );*/
/*	HANDLE_ERROR( cudaMemcpy(dev->ThPhi*/
/*								,host->ThPhi*/
/*								,Ntets*sizeof(int)*/
/*								,cudaMemcpyHostToDevice) );*/
/*	HANDLE_ERROR( cudaMemcpy(dev->nodeRank*/
/*								,host->nodeRank*/
/*								,Nnodes*sizeof(int)*/
/*								,cudaMemcpyHostToDevice) );*/
/*	HANDLE_ERROR( cudaMemcpy(dev->m*/
/*								,host->m*/
/*								,Nnodes*sizeof(real)*/
/*								,cudaMemcpyHostToDevice) );*/
/*	HANDLE_ERROR( cudaMemcpy(dev->TetVol*/
/*								,host->TetVol*/
/*								,Ntets*sizeof(real)*/
/*								,cudaMemcpyHostToDevice) );*/

/*	HANDLE_ERROR( cudaMemcpy(dev->S*/
/*								,host->S*/
/*								,Ntets*sizeof(int)*/
/*								,cudaMemcpyHostToDevice) );*/
/*	HANDLE_ERROR( cudaMemset( dev->L, 0, Ntets*sizeof(int) ) );*/
/*	HANDLE_ERROR( cudaMemcpy2D( dev->A*/
/*								, dev->Apitch*/
/*								, host->A*/
/*								, widthTETS*sizeof(real)*/
/*								, widthTETS*sizeof(real)*/
/*                                , height16*/
/*								, cudaMemcpyHostToDevice ) );*/
/*	HANDLE_ERROR( cudaMemcpy2D( dev->TetToNode*/
/*								, dev->TetToNodepitch*/
/*								, host->TetToNode*/
/*								, widthTETS*sizeof(int)*/
/*								, widthTETS*sizeof(int)*/
/*								, height4*/
/*								, cudaMemcpyHostToDevice ) );*/
/*	HANDLE_ERROR( cudaMemcpy2D( dev->r0*/
/*								, dev->r0pitch*/
/*								, host->r0*/
/*								, widthNODE*sizeof(real)*/
/*								, widthNODE*sizeof(real)*/
/*								, height3*/
/*								, cudaMemcpyHostToDevice ) );*/
/*	HANDLE_ERROR( cudaMemcpy2D( dev->r*/
/*								, dev->rpitch*/
/*								, host->r*/
/*								, widthNODE*sizeof(real)*/
/*								, widthNODE*sizeof(real)*/
/*								, height3*/
/*								, cudaMemcpyHostToDevice ) );*/
/*	HANDLE_ERROR( cudaMemcpy2D( dev->v*/
/*								, dev->vpitch*/
/*								, host->v*/
/*                                , widthNODE*sizeof(real)*/
/*								, widthNODE*sizeof(real)*/
/*                                , height3*/
/*								, cudaMemcpyHostToDevice ) );*/
/*	HANDLE_ERROR( cudaMemcpy2D( dev->F*/
/*								, dev->Fpitch*/
/*								, host->F*/
/*                                , widthNODE*sizeof(real)*/
/*								, widthNODE*sizeof(real)*/
/*                                , height3*/
/*								, cudaMemcpyHostToDevice ) );*/

	// put parameters into __constant__ memory
	PackedParameters _tmp;
	_tmp.Alpha = params->Material.Alpha;
	_tmp.Cxxxx = params->Material.Cxxxx;
	_tmp.Cxxyy = params->Material.Cxxyy;
	_tmp.Cxyxy = params->Material.Cxyxy;
	_tmp.Density = params->Material.Density;
	_tmp.Dt = params->Dynamics.Dt;
	_tmp.Damp = params->Dynamics.Damp;
	_tmp.Scale = params->Mesh.Scale;
	_tmp.SInitial = params->Actuation.OrderParameter.SInitial;
	_tmp.Smax = params->Actuation.OrderParameter.Smax;
	_tmp.Smin = params->Actuation.OrderParameter.Smin;
	_tmp.SRateOn = params->Actuation.OrderParameter.SRateOn;
	_tmp.SRateOff = params->Actuation.OrderParameter.SRateOff;
	_tmp.IncidentAngle = params->Actuation.Optics.IncidentAngle;
	
	HANDLE_ERROR( cudaMemcpyToSymbol(Parameters, &_tmp, sizeof(PackedParameters)) );


	//================================================
	//bind linear pitched memory to 2D texture
	//================================================


/*	HANDLE_ERROR( cudaBindTexture2D( &offset */
/*									,texRef_r0*/
/*									, dev->r0*/
/*									, texRef_r0.channelDesc*/
/*									, widthNODE*/
/*									, height3*/
/*									, dev->r0pitch) );*/
/*	texRef_r0.normalized = false;*/
/*	//texRef_r0.filterMode = cudaFilterModeLinear;*/
/*	HANDLE_ERROR( cudaBindTexture2D( &offset */
/*									, texRef_r*/
/*									, dev->r*/
/*									, texRef_r.channelDesc*/
/*									, widthNODE*/
/*									, height3*/
/*									, dev->rpitch) );*/
/*	texRef_r.normalized = false;*/
	//texRef_rA.filterMode = cudaFilterModeLinear;
	printf("\ndata sent to device\n");

}


#endif//__DATATODEVICE_H__
