#include "data_manager.h"
#include "errorhandle.h"
#include "kernel_constants.h"

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

bool DataManager::SetSimulationParameters(SimulationParameters *params)
{
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
	return true;
}
