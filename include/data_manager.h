#ifndef __DATA_MANAGER_H__
#define __DATA_MANAGER_H__

#include "datastruct.h"
#include "data_operations.h"
#include "data_procedures.h"
#include "simulation_parameters.h"

/// 
/// Controller class for data operations. Construct by injecting
/// Host and Device DataBlock pointers to be used in memory ops.
class DataManager
{
	///
	/// pointers to data
	HostDataBlock *host;
	DevDataBlock *dev;
	
	///
	/// pointers to data procedures
	DataProcedure * setup;
	DataProcedure * exit;
	DataProcedure * print;

public:

	///
	/// Construct with host and dev data blocks and data procedures
	/// needed to when running in SimulationRunner
	DataManager(HostDataBlock*, DevDataBlock*, DataProcedure*, DataProcedure*, DataProcedure *);

	///
	/// Pass a DataProcedure to be run with current object
	/// (DataProcedure: a logical grouping of DataOperations)
	bool Execute(DataProcedure*);

	///
	/// Pass a single DataOperation to be run with current object
	bool Execute(DataOperation*);

	///
	/// Pass in SimulationParameters to update on the gpu
	bool UpdateSimulationParameters(SimulationParameters*);

	///
	/// Pass in SimulationParameters to initially put them on gpu
	bool Setup(SimulationParameters*);

	///
	/// Executes the DataProcedure injected in contstrtor which
	/// does anything required to get print data on cpu
	bool GetPrintData();

	///
	/// Executes the DataProcedure injected in constructor which
	/// does anything required to run when destorying a SimulationRunner
	bool Exit();
};

#endif
