#include "data_manager.h"
#include "errorhandle.h"
#include "kernel_constants.h"

DataManager::DataManager(HostDataBlock *hostDataBlock, 
	DevDataBlock *devDataBlock,
	DataProcedure *setupProcedure,
	DataProcedure *printProcedure = NULL,
	DataProcedure *exitProcedure = NULL)
{
	this->host = hostDataBlock;
	this->dev = devDataBlock;
	this->setup = setupProcedure;
	this->print = printProcedure;
	this->exit = exitProcedure;
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

bool DataManager::UpdateSimulationParameters(SimulationParameters *params)
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


bool DataManager::Setup(SimulationParameters * parameters)
{
	bool p_success = this->UpdateSimulationParameters(parameters);	
	if (this->setup == NULL)
	{
		return p_success;
	}
	
	bool s_success = Execute(setup);
	return p_success && s_success;
}

bool DataManager::GetPrintData()
{
	if (this->print == NULL)
	{
		return true;
	}
	return Execute(print);
}

bool DataManager::Exit()
{
	if (this->exit == NULL)
	{
		return true;
	}
	return Execute(exit);
}
