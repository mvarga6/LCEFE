#ifndef __SIMULATION_RUNNER_H__
#define __SIMULATION_RUNNER_H__

#include "simulation_parameters.h"
#include "output_writer.h"
#include "data_manager.h"
#include "logger.h"
#include "performance_recorder.h"
#include "datastruct.h"
#include "experiment.h"
#include "physics_model.h"

///
/// Class that handles running a simulation.
/// Constains the simulation MAIN-LOOP
///
class SimulationRunner
{
	VtkWriter 			 * vtkWriter;
	Logger 				 * log;
	PerformanceRecorder	 * recorder;
	
public:
	SimulationRunner(VtkWriter*, 
		Logger *log,
		PerformanceRecorder*);
		
	///
	/// Executes the main loop: simulation dynamics
	///
	void RunDynamics(DataManager*, Physics*, SimulationParameters*, Experiment*);

	///
	/// Exits the simulation
	///
	int Exit();
};

#endif
