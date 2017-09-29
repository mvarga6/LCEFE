#include "data_manager.h"

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
