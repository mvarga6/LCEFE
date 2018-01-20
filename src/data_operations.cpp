#include "../include/data_operations.h"
#include "../include/errorhandle.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "../include/texdef.h"

size_t global_texture_offset = 0;
texture<real, 2, cudaReadModeElementType> texRef_r0;
texture<real, 2, cudaReadModeElementType> texRef_r;

bool PullPositionFromGpu::operator()(DevDataBlock *dev, HostDataBlock *host)
{
	size_t size = host->Nnodes*sizeof(real);
	HANDLE_ERROR( cudaMemcpy2D(  host->r
								, size
								, dev->r
								, dev->rpitch
								, size
								, 3
								, cudaMemcpyDeviceToHost ) );
	return true;
}


bool PullVelocityFromGpu::operator()(DevDataBlock *dev, HostDataBlock *host)
{
	size_t size = host->Nnodes*sizeof(real);
	HANDLE_ERROR( cudaMemcpy2D(  host->v
								, size
								, dev->v
								, dev->vpitch
								, size
								, 3
								, cudaMemcpyDeviceToHost ) );
	return true;
}


bool PullForceFromGpu::operator()(DevDataBlock *dev, HostDataBlock *host)
{
	size_t size = host->Nnodes*sizeof(real);
	HANDLE_ERROR( cudaMemcpy2D(  host->F
								, size
								, dev->F
								, dev->Fpitch
								, size
								, 3
								, cudaMemcpyDeviceToHost ) );
	return true;
}


bool PullEnergyFromGpu::operator()(DevDataBlock *dev, HostDataBlock *host)
{
	size_t size = host->Ntets*sizeof(real);
	HANDLE_ERROR( cudaMemcpy(  host->pe
								, dev->pe
								, size
								, cudaMemcpyDeviceToHost ) );
	return true;
}


bool PushTetNodeRankToGpu::operator()(DevDataBlock *dev, HostDataBlock *host)
{
	// HANDLE_ERROR( 
	// 	cudaMemcpy(dev->TetNodeRank
	// 	,host->TetNodeRank
	// 	,host->Ntets*4*sizeof(int)
	// 	,cudaMemcpyHostToDevice) 
	// );
	// return true;

	HANDLE_ERROR( 
		cudaMemcpy2D( dev->TetNodeRank
		, dev->TetNodeRankpitch
		, host->TetNodeRank
		, host->Ntets*sizeof(int)
		, host->Ntets*sizeof(int)
        , 4
		, cudaMemcpyHostToDevice ) 
	);
	return true;
}

bool PushTriNodeRankToGpu::operator()(DevDataBlock *dev, HostDataBlock *host)
{
	HANDLE_ERROR( 
		cudaMemcpy2D( dev->TriNodeRank
		, dev->TriNodeRankpitch
		, host->TriNodeRank
		, host->Ntris*sizeof(int)
		, host->Ntris*sizeof(int)
        , 3
		, cudaMemcpyHostToDevice ) 
	);
	return true;
}


bool PushThetaPhiToGpu::operator()(DevDataBlock *dev, HostDataBlock *host)
{
	HANDLE_ERROR( 
		cudaMemcpy(dev->ThPhi
		,host->ThPhi
		,host->Ntets*sizeof(int)
		,cudaMemcpyHostToDevice) 
	);
	return true;	
}


bool PushNodeRankToGpu::operator()(DevDataBlock *dev, HostDataBlock *host)
{
	HANDLE_ERROR( 
		cudaMemcpy(dev->nodeRank
		,host->nodeRank
		,host->Nnodes*sizeof(int)
		,cudaMemcpyHostToDevice) 
	);
	return true;	
}


bool PushMassToGpu::operator()(DevDataBlock *dev, HostDataBlock *host)
{
	HANDLE_ERROR( 
		cudaMemcpy(dev->m
		,host->m
		,host->Nnodes*sizeof(real)
		,cudaMemcpyHostToDevice) 
	);
	return true;	
}


bool PushTetVolumeToGpu::operator()(DevDataBlock *dev, HostDataBlock *host)
{
	HANDLE_ERROR( 
		cudaMemcpy(dev->TetVol
		,host->TetVol
		,host->Ntets*sizeof(real)
		,cudaMemcpyHostToDevice) 
	);
	return true;	
}


bool PushNematicOrderParameterToGpu::operator()(DevDataBlock *dev, HostDataBlock *host)
{
	HANDLE_ERROR( 
		cudaMemcpy(dev->S
		,host->S
		,host->Ntets*sizeof(int)
		,cudaMemcpyHostToDevice) 
	);
	return true;	
}


bool PushAinvToGpu::operator()(DevDataBlock *dev, HostDataBlock *host)
{
	HANDLE_ERROR( 
		cudaMemcpy2D( dev->A
		, dev->Apitch
		, host->A
		, host->Ntets*sizeof(real)
		, host->Ntets*sizeof(real)
        , 16
		, cudaMemcpyHostToDevice ) 
	);
	return true;	
}


bool PushTetToNodeMapToGpu::operator()(DevDataBlock *dev, HostDataBlock *host)
{
	HANDLE_ERROR( 
		cudaMemcpy2D( dev->TetToNode
		, dev->TetToNodepitch
		, host->TetToNode
		, host->Ntets*sizeof(int)
		, host->Ntets*sizeof(int)
        , 4
		, cudaMemcpyHostToDevice ) 
	);
	return true;	
}

bool PushTriToNodeMapToGpu::operator()(DevDataBlock *dev, HostDataBlock *host)
{
	HANDLE_ERROR( 
		cudaMemcpy2D( dev->TriToNode
		, dev->TriToNodepitch
		, host->TriToNode
		, host->Ntris*sizeof(int)
		, host->Ntris*sizeof(int)
        , 3
		, cudaMemcpyHostToDevice ) 
	);
	return true;
}

bool PushPostionToGpu::operator()(DevDataBlock *dev, HostDataBlock *host)
{
	HANDLE_ERROR( 
		cudaMemcpy2D( dev->r
		, dev->rpitch
		, host->r
		, host->Nnodes*sizeof(real)
		, host->Nnodes*sizeof(real)
        , 3
		, cudaMemcpyHostToDevice ) 
	);
	return true;	
}


bool PushReferencePositionToGpu::operator()(DevDataBlock *dev, HostDataBlock *host)
{
	HANDLE_ERROR( 
		cudaMemcpy2D( dev->r0
		, dev->r0pitch
		, host->r0
		, host->Nnodes*sizeof(real)
		, host->Nnodes*sizeof(real)
        , 3
		, cudaMemcpyHostToDevice ) 
	);
	return true;	
}


bool PushVelocityToGpu::operator()(DevDataBlock *dev, HostDataBlock *host)
{
	HANDLE_ERROR( 
		cudaMemcpy2D( dev->v
		, dev->vpitch
		, host->v
		, host->Nnodes*sizeof(real)
		, host->Nnodes*sizeof(real)
        , 3
		, cudaMemcpyHostToDevice ) 
	);
	return true;	
}


bool PushForceToGpu::operator()(DevDataBlock *dev, HostDataBlock *host)
{
	HANDLE_ERROR( 
		cudaMemcpy2D( dev->F
		, dev->Fpitch
		, host->F
		, host->Nnodes*sizeof(real)
		, host->Nnodes*sizeof(real)
        , 3
		, cudaMemcpyHostToDevice ) 
	);
	return true;	
}


bool PushTriAreaToGpu::operator()(DevDataBlock *dev, HostDataBlock *host)
{
	HANDLE_ERROR( 
		cudaMemcpy(dev->TriArea
		,host->TriArea
		,host->Ntris*sizeof(real)
		,cudaMemcpyHostToDevice) 
	);
	return true;
}


bool PushTriNormalToGpu::operator()(DevDataBlock *dev, HostDataBlock *host)
{
	HANDLE_ERROR( 
		cudaMemcpy2D( dev->TriNormal
		, dev->TriNormalpitch
		, host->TriNormal
		, host->Ntris*sizeof(real)
		, host->Ntris*sizeof(real)
        , 3
		, cudaMemcpyHostToDevice ) 
	);

	HANDLE_ERROR( 
		cudaMemcpy( dev->TriNormalSign
		, host->TriNormalSign
		, host->Ntris*sizeof(int)
		, cudaMemcpyHostToDevice ) 
	);

	return true;
}


bool BindPositionTexture::operator()(DevDataBlock *dev, HostDataBlock *host)
{
	HANDLE_ERROR( 
		cudaBindTexture2D( &global_texture_offset 
		, texRef_r
		, dev->r
		, texRef_r.channelDesc
		, host->Nnodes
		, 3
		, dev->rpitch) );
	texRef_r.normalized = false;
	return true;	
}


bool BindReferencePositionTexture::operator()(DevDataBlock *dev, HostDataBlock *host)
{
	HANDLE_ERROR( 
		cudaBindTexture2D( &global_texture_offset 
		, texRef_r0
		, dev->r0
		, texRef_r0.channelDesc
		, host->Nnodes
		, 3
		, dev->r0pitch) );
	texRef_r0.normalized = false;
	return true;	
}




