#include "data_operations.h"
#include "errorhandle.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "texdef.h"

size_t global_texture_offset = 0;
texture<real, 2, cudaReadModeElementType> texRef_r0;
texture<real, 2, cudaReadModeElementType> texRef_r;

bool PullPositionFromGpu::operator()(DevDataBlock *dev, HostDataBlock *host)
{
	size_t size = host->Nnodes*3*sizeof(real);
	HANDLE_ERROR( cudaMemcpy(  host->r
							 , dev->r
							 , size
							 , cudaMemcpyDeviceToHost ) );
	return true;
}


bool PullVelocityFromGpu::operator()(DevDataBlock *dev, HostDataBlock *host)
{
	size_t size = host->Nnodes*3*sizeof(real);
	HANDLE_ERROR( cudaMemcpy(  host->v
							 , dev->v
							 , size
							 , cudaMemcpyDeviceToHost ) );
	return true;
}


bool PullForceFromGpu::operator()(DevDataBlock *dev, HostDataBlock *host)
{
	size_t size = host->Nnodes*3*sizeof(real);
	HANDLE_ERROR( cudaMemcpy(  host->F
								, dev->F
								, size
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
	HANDLE_ERROR( 
		cudaMemcpy(dev->TetNodeRank
		,host->TetNodeRank
		,host->Ntets*4*sizeof(int)
		,cudaMemcpyHostToDevice) 
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
	size_t size = host->Ntets*16*sizeof(real);
	HANDLE_ERROR( 
		cudaMemcpy( dev->A
					, host->A
					, size
					, cudaMemcpyHostToDevice ) 
	);
	return true;	
}


bool PushTetToNodeMapToGpu::operator()(DevDataBlock *dev, HostDataBlock *host)
{
	size_t size = host->Ntets*4*sizeof(int);
	HANDLE_ERROR( 
		cudaMemcpy( dev->TetToNode
				, host->TetToNode
				, size
				, cudaMemcpyHostToDevice ) 
	);
	return true;	
}


bool PushPostionToGpu::operator()(DevDataBlock *dev, HostDataBlock *host)
{
	size_t size = host->Nnodes*3*sizeof(real);
	HANDLE_ERROR( 
		cudaMemcpy( dev->r
				  , host->r
				  , size
				  , cudaMemcpyHostToDevice ) 
	);
	return true;	
}


bool PushReferencePositionToGpu::operator()(DevDataBlock *dev, HostDataBlock *host)
{
	size_t size = host->Nnodes*3*sizeof(real);
	HANDLE_ERROR( 
		cudaMemcpy( dev->r0
				  , host->r0
				  , size
				  , cudaMemcpyHostToDevice ) 
	);
	return true;	
}


bool PushVelocityToGpu::operator()(DevDataBlock *dev, HostDataBlock *host)
{
	size_t size = host->Nnodes*3*sizeof(real);
	HANDLE_ERROR( 
		cudaMemcpy( dev->v
				  , host->v
				  , size
				  , cudaMemcpyHostToDevice ) 
	);
	return true;	
}


bool PushForceToGpu::operator()(DevDataBlock *dev, HostDataBlock *host)
{
	size_t size = host->Nnodes*3*sizeof(real);
	HANDLE_ERROR( 
		cudaMemcpy( dev->F
				  , host->F
				  , size
				  , cudaMemcpyHostToDevice ) 
	);
	return true;	
}


// bool BindPositionTexture::operator()(DevDataBlock *dev, HostDataBlock *host)
// {
// 	HANDLE_ERROR( 
// 		cudaBindTexture2D( &global_texture_offset 
// 		, texRef_r
// 		, dev->r
// 		, texRef_r.channelDesc
// 		, host->Nnodes
// 		, 3
// 		, dev->rpitch) );
// 	texRef_r.normalized = false;
// 	return true;	
// }


// bool BindReferencePositionTexture::operator()(DevDataBlock *dev, HostDataBlock *host)
// {
// 	HANDLE_ERROR( 
// 		cudaBindTexture2D( &global_texture_offset 
// 		, texRef_r0
// 		, dev->r0
// 		, texRef_r0.channelDesc
// 		, host->Nnodes
// 		, 3
// 		, dev->r0pitch) );
// 	texRef_r0.normalized = false;
// 	return true;	
// }