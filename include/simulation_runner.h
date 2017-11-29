#ifndef __SIMULATION_RUNNER_H__
#define __SIMULATION_RUNNER_H__

#include "simulation_parameters.h"
#include "output_writer.h"
#include "data_manager.h"
#include "logger.h"
#include "performance_recorder.h"
#include "datastruct.h"

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
		
	void RunDynamics();
	int Exit();
};

#endif
