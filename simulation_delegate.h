#ifndef __SIMULATION_RUNNER_H__
#define __SIMULATION_RUNNER_H__

#include "simulation_parameters.h"
#include "parameters_reader.h"
#include "parameters_writer.h"
#include "output_writer.h"
#include "data_manager.h"
#include "performance_recorder.h"
#include "director_field.h"
#include "mesh.h"
#include "logger.h"

using namespace std;

enum class ProcessInputResult : int
{
	COMPLETE = 0,
	INITIALIZE_ONLY = 1,
	TEST = 2,
	INVALID_INPUT = 3
};
	
enum class InitializeResult : int
{
	SUCCESS = 1,
	TIMEOUT = 2,
	FAILURE = 3
};

class SimulationDelegate
{
	// Operational stuff
	SimulationParameters * parameters;
	Logger 				 * logger;
	PerformanceRecorder  * performance;
	VtkWriter			 * vtkWriter;
	
	// Data
	Mesh 				 * mesh;
	DirectorField		 * director;
	DevDataBlock		 * dev;
	HostDataBlock		 * host;
	DataManager			 * dataManager;
	vector<int>			   surfaceTetraIds;
	
public:

	/*
	  Create a delegate for running simulations
	*/
	SimulationDelegate();
	
	/*
	  Gather all input required for simulation
	*/
	ProcessInputResult ProcessUserInput(int argc, char *argv[]);	
	
	/*
	  Initializes components required to run a simulation.
	*/
	InitializeResult Initialize();
	
	/*
	  Runs a simulation in the current state.
	*/
	int Run();
};

#endif
