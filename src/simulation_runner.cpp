#include "simulation_runner.h"
#include "anyerrors.h"
#include "exit_program.h"
#include "printVTKframe.h"
#include "physics_model.h"
#include "experiment.h"

SimulationRunner::SimulationRunner(
	VtkWriter *vtkWriter,
	Logger *log,
	PerformanceRecorder *recorder)
{
	this->vtkWriter = vtkWriter;
	this->log = log;
	this->recorder = recorder;
}


void SimulationRunner::RunDynamics(DataManager* data, Physics *physics, SimulationParameters* parameters, Experiment* experiment)
{
	// setup on data
	data->Setup();

	// initalize data to specific physics
	physics->Initialize(data);
	
	// data blocks
	DevDataBlock * dev = data->DeviceData();
	HostDataBlock * host = data->HostData();

	// local variables
	const real dt 			= parameters->Dynamics.Dt;
//	const real meshScale 	= parameters->Mesh.Scale;
	const int iterPerFrame 	= parameters->Output.FrameRate;
	const int nSteps 		= parameters->Dynamics.Nsteps;

	// the lab frame time
	real t = 0;

	// the main simulation time loop
	recorder->Create("time-loop")->Start();
	this->log->Msg("Beginning Dynamics");
	for(int iKern = 0; iKern < nSteps; iKern++)
	{
		// calculate the current time
		t = dt*real(iKern);

		// calculate force and send force components to be summed
		//ForceKernel<<<BlocksTet,Threads_Per_Block>>>(*dev, dt*real(iKern));
		physics->CalculateForces(data, t);

		// sum forces and update positions	
		//UpdateKernel<<<BlocksNode,Threads_Per_Block>>>(*dev);
		physics->UpdateSystem(data);

		// updates related the the experiment
		experiment->Update(dt);

		// sync threads before updating
		cudaThreadSynchronize();
  		recorder->Mark("time-loop");

		// pull data to host then print to files
		if((iKern) % iterPerFrame == 0)
		{
			printf("\n==============================================");
			printf("\nKernel: %d of %d", iKern + 1, nSteps);
			printf("\nTime: %f seconds", real(iKern)*dt);

			recorder->Log("time-loop");
			
			// execute procedure using
			data->GetPrintData();
		
			//print frame
			printVTKframe(dev
				,host
				,parameters->Output.Base
				,iKern+1);
		}
		
	}//iKern
}


int SimulationRunner::Exit()
{
	//this->dataManager->Exit();
	any_errors();
	//exit_program(this->dev);
	return 0;
}
