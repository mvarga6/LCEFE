#include "simulation_delegate.h"
#include "simulation_parameters.h"
#include "parameters_reader.h"
#include "logger.h"

SimulationDelegate::SimulationDelegate()
{
	// alloc private variables
	parameters = new SimulationParameters();	
	performance = new PerformanceRecorder();
}

ProcessInputResult SimulationDelegate::ProcessUserInput(int argc, char *argv[])
{
	performance->Create("UserInput")->Start();

	// Get parameters from command line
	ParametersReader * reader = new ParametersReader();
	
	// read cmdline	for parameters file	
	ParseResult result = reader->ReadFromCmdline(argc, argv, *parameters);
	if (result != ParseResult::SUCCESS)
	{
		Logger::Default->Msg(ParametersReader::GetErrorMsg(result));
		return ProcessInputResult::INVALID_INPUT;
	}
	
	// from file if given
	if (!parameters->File.empty())
	{
		result = reader->ReadFromFile(parameters->File, *parameters);
		if (result != ParseResult::SUCCESS)
		{
			Logger::Default->Msg(ParametersReader::GetErrorMsg(result));
			return ProcessInputResult::INVALID_INPUT;
		}
	}
	
	// override parameter files values with cmdline values
	result = reader->ReadFromCmdline(argc, argv, *parameters);
	if (result != ParseResult::SUCCESS)
	{
		Logger::Default->Msg(ParametersReader::GetErrorMsg(result));
		return ProcessInputResult::INVALID_INPUT;
	}
	
	// switch on loggertype from usering input
	switch(parameters->Output.LogType)
	{
	case LoggerType::NONE:
		printf("Logging disabled");
		break;
		
	case LoggerType::CONSOLE:
		logger = new ConsoleLogger(parameters->Output.LogLevel);
		break;
		
	case LoggerType::FILE:
		logger = new ConsoleLogger(parameters->Output.LogLevel);
		logger->Msg("File logger not implemented! Converting to console logger.", LogEntryPriority::WARNING);
		break;
	}	
	
	//
	// Process user input for director field
	//
	
	// Stop performance tracking on this step
	performance->Stop("UserInput");	
	
	return ProcessInputResult::TEST;
}


InitializeResult SimulationDelegate::Initialize()
{
	// create the mesh
	this->mesh = new Mesh(parameters, logger);
	
	// load the mesh
	bool cachedMesh;
	if (!mesh->Load(&cachedMesh))
	{
		// log failure
		exit(10);
	}
	
	// we haven't cached an optimized version of the mesh
	if (!cachedMesh)
	{
		// optimize the mesh
		MeshOptimizer * simpleSort = new SortOnTetrahedraPosition();
		MeshOptimizer * mcReorder = new MonteCarloMinimizeDistanceBetweenPairs(300, 0.01f, 0.99999f);
		MeshOptimizer * reIndex = new ReassignIndices();
		mesh->Apply(simpleSort);
		mesh->Apply(mcReorder);
		mesh->Apply(reIndex);
		
		// save the optimized mesh
		mesh->Cache();
	}
	
	return InitializeResult::SUCCESS;
}


int SimulationDelegate::Run()
{
	return 1;
}
