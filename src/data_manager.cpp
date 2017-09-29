#include "data_manager.h"
#include "errorhandle.h"
#include "cuda.h"
#include "cuda_runtime.h"


bool PullPositionFromGpu::operator()(DevDataBlock *dev, HostDataBlock *host)
{
	size_t size = host->Nnodes*sizeof(float);
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
	size_t size = host->Nnodes*sizeof(float);
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
	size_t size = host->Nnodes*sizeof(float);
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
	size_t size = host->Ntets*sizeof(float);
	HANDLE_ERROR( cudaMemcpy(  host->pe
								, dev->pe
								, size
								, cudaMemcpyDeviceToHost ) );
	return true;
}

DataManager::DataManager(HostDataBlock *hostDataBlock, DevDataBlock *devDataBlock)
{
	this->host = hostDataBlock;
	this->dev = devDataBlock;
}


bool DataManager::Execute(DataProcedure *procedure)
{
	bool success = true;

	for(auto operation : procedure->Operations)
	{
		success = Execute(operation) && success;
	}
	
	return success;
}


bool DataManager::Execute(DataOperation *operation)
{
	return (*operation)(dev, host);
}
