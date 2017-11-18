#include "simulation_runner.h"
#include "anyerrors.h"
#include "exit_program.h"
#include "printVTKframe.h"
#include "physics_kernels.h"

SimulationRunner::SimulationRunner(
	SimulationParameters *parameters, 
	VtkWriter *vtkWriter, 
	DataManager *dataManager,
	Logger *log,
	PerformanceRecorder *recorder,
	HostDataBlock *host,
	DevDataBlock *dev)
{
	this->parameters = parameters;
	this->vtkWriter = vtkWriter;
	this->dataManager = dataManager;
	this->log = log;
	this->recorder = recorder;
	this->host = host;
	this->dev = dev;
}


void SimulationRunner::RunDynamics()
{
	// TODO: DataProcdure for getting data should be inject somehow
	// Prossibly into the printer, so it know's how to get data it needs.
	//DataProcedure *getPrintData = new GetPrintData();
	this->dataManager->Setup(parameters);
	
	// local variables
	const real dt 			= this->parameters->Dynamics.Dt;
	const real meshScale 	= this->parameters->Mesh.Scale;
	const int iterPerFrame 	= this->parameters->Output.FrameRate;
	const int nSteps 		= this->parameters->Dynamics.Nsteps;
	const int Ntets  		= this->dev->Ntets;
	const int Nnodes  		= this->dev->Nnodes;
	
	// calculate how to run on gpu
	const int Threads_Per_Block = this->parameters->Gpu.ThreadsPerBlock;
	const int BlocksTet = (Ntets + Threads_Per_Block) / Threads_Per_Block;
	const int BlocksNode = (Nnodes + Threads_Per_Block) / Threads_Per_Block;
	
	recorder->Create("time-loop")->Start();
	this->log->Msg("Beginning Dynamics");
	
	// the main simulation time loop
	for(int iKern = 0; iKern < nSteps; iKern++)
	{
		// calculate force and send force components to be summed
		ForceKernel<<<BlocksTet,Threads_Per_Block>>>(*dev, dt*real(iKern));

		// sum forces and update positions	
		UpdateKernel<<<BlocksNode,Threads_Per_Block>>>(*dev);

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
			//dataManager->Execute(getPrintData);
			this->dataManager->GetPrintData();
		
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
	this->dataManager->Exit();
	any_errors();
	exit_program(this->dev);
}
