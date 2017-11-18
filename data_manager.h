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
	// pointers to data
	HostDataBlock *host;
	DevDataBlock *dev;
	
	// pointers to data procedures
	DataProcedure * setup;
	DataProcedure * exit;
	DataProcedure * print;

public:
	DataManager(HostDataBlock*, DevDataBlock*, DataProcedure*, DataProcedure*, DataProcedure *);
	bool Execute(DataProcedure*);
	bool Execute(DataOperation*);
	bool UpdateSimulationParameters(SimulationParameters*);
	bool Setup(SimulationParameters*);
	bool GetPrintData();
	bool Exit();
};

#endif
