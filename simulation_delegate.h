#ifndef __SIMULATION_RUNNER_H__
#define __SIMULATION_RUNNER_H__

#include "simulation_parameters.h"
#include "logger.h"
#include "performance_recorder.h"
#include "mesh.h"

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
	SimulationParameters * parameters;
	Logger 				 * logger;
	PerformanceRecorder  * performance;
	Mesh 				 * mesh;
	
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
