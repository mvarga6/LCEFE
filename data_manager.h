#ifndef __DATA_MANAGER_H__
#define __DATA_MANAGER_H__

#include "datastruct.h"
#include "data_operations.h"
#include "data_procedures.h"
#include "simulation_parameters.h"

/*
  Controller class for data operations. Construct by injecting
  Host and Device DataBlock pointers to be used in memory ops.
*/
class DataManager
{
	HostDataBlock *host;
	DevDataBlock *dev;

public:
	DataManager(HostDataBlock *hostDataBlock, DevDataBlock *devDataBlock);
	bool Execute(DataProcedure *procedure);
	bool Execute(DataOperation *operation);
	bool SetSimulationParameters(SimulationParameters *parameters);
};

#endif
