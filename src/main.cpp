//=============================================================//
//                                                             //
//            ||Gpu Accelerated Fineite Element ||             //
//                                                             //
//              --------Version 6.0----------                  //
//                                                             //
//                                                             //
//                                                             //
//    Authors: Andrew Konya      (Kent State University)       //
//             Robin Selinger    (Kent State University)       // 
//             Badel MBanga      (kent State University)       //
//                                                             //
//   Finite elemnt simulation executed on GPU using CUDA       //
//   Hybrid MD finite element algorithm used to allow          //
//   all computations be implemented locally requireing        //
//   parallelization of all prccess in calculation             //
//                                                             //
//=============================================================//


//#include "mainhead.h"
#include "simulation_parameters.h"
#include "parameters_reader.h"
#include "parameters_writer.h"
#include "output_writer.h"
#include "data_manager.h"
#include "performance_recorder.h"
#include "mesh_operations.h"
#include "director_field.h"
#include "mesh.h"
#include "functions.hpp"
#include "logger.h"
#include "helpers_math.h"
#include "data_procedures.h"
#include "simulation_runner.h"

// these will go away into their own service class
//#include "rundynamics.h"
//#include "getAs.h"
//#include "setn.h"
//#include "printmeshorder.h"
//#include "packdata.h"
#include "errorhandle.h"
//#include "datatodevice.h"
//#include "anyerrors.h"
//#include "exit_program.h"

int main(int argc, char *argv[])
{
	// Read simulation parameters
	SimulationParameters parameters;
	ParametersReader * reader = new ParametersReader();
	ParseResult result;
	
	// from cmdline
	result = reader->ReadFromCmdline(argc, argv, parameters);
	if (result != SUCCESS)
	{
		return (int)result;
	}
		
	// from file if given
	if (!parameters.File.empty())
	{
		result = reader->ReadFromFile(parameters.File, parameters);
		if (result != SUCCESS)
		{
			return (int)result;
		}
	}
	
	// to write the parameters to console
	ParametersWriter * writer = new ConsoleWriter();
	writer->Write(parameters);
		
	// create a console logger
	Logger * log = new ConsoleLogger();
	
	// for timing data
	PerformanceRecorder * recorder = new PerformanceRecorder();;
	recorder->Create("init")->Start();
	
	// for printing to output files
	VtkWriter * vtkWriter = new VtkWriter(parameters.Output.Base);
	
	// the mesh object
	Mesh * mesh = new Mesh(&parameters, log);

	bool cachedMesh;
	if (!mesh->Load(&cachedMesh))
	{
		// TODO: log failure
		exit(10);
	}

	// we cache an optimized version of the mesh
	if (!cachedMesh)
	{
		// optimize the mesh
		log->Msg(" *** Optimizing mesh *** ");
		
		// simple sorting based on location in sim space
		mesh->Apply(new SortOnTetrahedraPosition());
		
		// re-order using mc simulation
		mesh->Apply(new MonteCarloMinimizeDistanceBetweenPairs(300.0f, 0.01f, 0.999f));
		
		// re-index the mesh and tet's neighbors
		mesh->Apply(new ReassignIndices());
		
		// save the optimized mesh
		mesh->Cache();
	}
	else
	{
		mesh->Apply(new ReassignIndices());
		log->Msg("Mesh Loaded from cache!");
	}
	
	// create director field
	const float3 origin = make_float3(0.0f, 0.0f, 0.0f);
	DirectorField * director = new RadialDirectorField(origin);
	
	mesh->Apply(new CalculateVolumes());
	mesh->Apply(new CalculateAinv());
	mesh->Apply(new SetDirector(director));
			
	
	//pritn director
	mesh->Tets->printDirector(parameters.Output.Base);

	// Create Host and Device Data blocks with the mesh
	HostDataBlock * host = new HostDataBlock(mesh->Nodes, mesh->Tets, &parameters);
	DevDataBlock * dev = host->CreateDevDataBlock();
	
	DataProcedure * setup = new PushAllToGpu();
	DataProcedure * print = new GetPrintData();
	DataManager * dataManager = new DataManager(host, dev, setup, print);

	//Print Simulation Parameters and Such
	printf("\n\nPrepared for dynamics with:\nsteps/frame: %d\nVolume: %f cm^3\nMass: %f kg\n",
				parameters.Output.FrameRate,
				host->totalVolume,
				host->totalVolume * parameters.Material.Density);


	// TODO: Move gpu info print somewhere else
	//Get Device properties
	cudaDeviceProp prop;
	HANDLE_ERROR(cudaGetDeviceProperties(&prop,0));
	printf( "Code executing on %s\n\n", prop.name );
	//displayGPUinfo(prop);
	 
	recorder->Stop("init");
	recorder->Log("init");
	 
	//=================================================================
	//run dynamics
	//=================================================================
	//run_dynamics(dev, host, &parameters, vtkWriter, dataManager, recorder);	
	//check for CUDA erros
	//any_errors();
	//exit_program(dev);
	
	SimulationRunner * sim = new SimulationRunner(
		&parameters,
		vtkWriter,
		dataManager,
		log,
		recorder,
		host,
		dev
	);
	
	sim->RunDynamics();
    return sim->Exit();
}
