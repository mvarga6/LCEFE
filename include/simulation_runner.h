#ifndef __SIMULATION_RUNNER_H__
#define __SIMULATION_RUNNER_H__

#include "simulation_parameters.h"
#include "output_writer.h"
#include "data_manager.h"
#include "logger.h"
#include "performance_recorder.h"
#include "datastruct.h"

///
/// Class that handles running a simulation.
/// Constains the simulation MAIN-LOOP
class SimulationRunner
{
	SimulationParameters * parameters;
	VtkWriter 			 * vtkWriter;
	DataManager			 * dataManager;
	Logger 				 * log;
	PerformanceRecorder	 * recorder;
	HostDataBlock		 * host;
	DevDataBlock		 * dev;
	
public:
	SimulationRunner(SimulationParameters*, 
		VtkWriter*, 
		DataManager*,
		Logger *log,
		PerformanceRecorder*,
		HostDataBlock*,
		DevDataBlock*);
		
	///
	/// Executes the main loop: simulation dynamics
	void RunDynamics();

	///
	/// Exits the simulation
	int Exit();
};

#endif
