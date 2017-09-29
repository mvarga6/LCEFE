#ifndef __DATA_MANAGER_H__
#define __DATA_MANAGER_H__

#include "datastruct.h"
#include <vector>

/*
  Abstract parent of anything that moves data to/from gpu
*/
class DataOperation { public: virtual bool operator()(DevDataBlock*, HostDataBlock*) = 0; };

/*
  DataOperation childen that act as they're names' suggest
*/
class PullPositionFromGpu : public DataOperation { public: virtual bool operator()(DevDataBlock*, HostDataBlock*); };
class PullVelocityFromGpu : public DataOperation { public: virtual bool operator()(DevDataBlock*, HostDataBlock*); };
class PullForceFromGpu : public DataOperation { public: virtual bool operator()(DevDataBlock*, HostDataBlock*); };
class PullEnergyFromGpu : public DataOperation { public: virtual bool operator()(DevDataBlock*, HostDataBlock*); };

/*
  A storage container for list of DataOperations, logically 
  grouped and ordered into a Procedure.  
  
  For custom precudures, define a new DataProcedure and
  manually add DataOperations to the Operations list.
*/
class DataProcedure
{
public:
	std::vector<DataOperation*> Operations;
};

/*
  An implementation of a DataProcedure that initiallizes
  itself to get data required to print.
*/
class GetPrintData : public DataProcedure
{
public:
	GetPrintData()
	{
		Operations.push_back(new PullPositionFromGpu());
		Operations.push_back(new PullVelocityFromGpu());
		Operations.push_back(new PullEnergyFromGpu());
	}
};

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
};

#endif
