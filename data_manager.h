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
	DataManager(HostDataBlock*, DevDataBlock*);
	bool Execute(DataProcedure*);
	bool Execute(DataOperation*);
	bool SetSimulationParameters(SimulationParameters*);
};

#endif
