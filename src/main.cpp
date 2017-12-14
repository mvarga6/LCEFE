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
#include "experiment.h"
#include "pointer.h"

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
	// create a console logger
	Logger * log = new ConsoleLogger();

	///
	/// Handle user input
	///

	// create parameters object
	ParametersReader * reader = new ParametersReader();
	SimulationParameters * parameters = reader->Read(argc, argv);

	if (!reader->Success())
	{
		printf("%s\n", reader->Result().Message.c_str());
		return reader->Status();
	}

	// SimulationParameters parameters = SimulationParameters::Default();
	
	// // from cmdline
	// ParseResult result = reader->ReadFromCmdline(argc, argv, parameters);
	// if (result != SUCCESS)
	// {
	// 	return (int)result;
	// }
		
	// // from file if given
	// if (!parameters.File.empty())
	// {
	// 	result = reader->ReadFromFile(parameters.File, parameters);
	// 	if (result != SUCCESS)
	// 	{
	// 		return (int)result;
	// 	}
	// }
	
	// to write the parameters to console
	ParametersWriter * writer = new ConsoleWriter();
	writer->Write(parameters);
	
	// for timing data
	PerformanceRecorder * recorder = new PerformanceRecorder();;
	recorder->Create("init")->Start();
	
	// for printing to output files
	VtkWriter * vtkWriter = new VtkWriter(parameters->Output.Base);
	
	///
	/// Create a Mesh from file
	///

	Mesh * mesh = new Mesh(parameters, log);

	bool cachedMesh;
	if (!mesh->Load(&cachedMesh))
	{
		log->Error("Failed to load mesh, exiting");
		exit(10);
	}

	///
	/// Optimize the mesh if need be
	///

	// we cache an optimized version of the mesh
	if (!cachedMesh)
	{
		// optimize the mesh
		log->Msg(" *** Optimizing mesh *** ");
		
		// simple sorting based on location in sim space
		mesh->Apply(new SortOnTetrahedraPosition());
		
		// re-order using mc simulation
		mesh->Apply(new MonteCarloMinimizeDistanceBetweenPairs(10000.0f, 0.001f, 0.999999f));
		
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
	
	///
	/// Calculate values for mesh
	///

	mesh->Apply(new CalculateVolumes());
	mesh->Apply(new CalculateAinv());

	///
	/// Endow mesh with liquid crystal properities
	///

	// create director field
	const float3 origin 	 = make_float3(0.0f, 0.0f, 0.0f);
	DirectorField * director = new RadialDirectorField(origin);
	mesh->Apply(new SetDirector(director));
			
	//print director
	mesh->Tets->printDirector(parameters->Output.Base);

	///
	/// Create data management objects
	///

	// Create Host and Device Data blocks with the mesh
	HostDataBlock 	* host 	= new HostDataBlock(mesh->Nodes, mesh->Tets, parameters);
	DevDataBlock 	* dev 	= host->CreateDevDataBlock();
	DataProcedure 	* setup = new PushAllToGpu();
	DataProcedure 	* print = new GetPrintData();
	DataManager 	* data 	= new DataManager(host, dev, parameters, setup, print, NULL);

	///
	/// Create the experiment to run
	///

	Experiment * experiment = new Experiment();
	real start = parameters->Dynamics.ExperimentStart();
	real stop = parameters->Dynamics.ExperimentStop();
	ExperimentComponent * orderDynamics = new NematicToIsotropic(start, stop, dev->HandleForS());
	experiment->AddComponent("OrderDynamics", orderDynamics);

	///
	/// Create the physics model to simulate
	///

	Physics * physics = new SelingerPhysics();

	///
	/// Print info before running simulation
	///

	//Print Simulation Parameters and Such
	// printf("\n\nPrepared for dynamics with:\nsteps/frame: %d\nVolume: %f cm^3\nMass: %f kg\n",
	// 			parameters.Output.FrameRate,
	// 			host->totalVolume,
	// 			host->totalVolume * parameters.Material.Density);


	// // TODO: Move gpu info print somewhere else
	// //Get Device properties
	// cudaDeviceProp prop;
	// HANDLE_ERROR(cudaGetDeviceProperties(&prop,0));
	// printf( "Code executing on %s\n\n", prop.name );
	//displayGPUinfo(prop);
	 
	recorder->Stop("init");
	recorder->Log("init");
	 
	///
	/// Run a simulation the experiment with given physics
	///

	// Create the simulation running environment
	SimulationRunner * sim = new SimulationRunner(vtkWriter, log, recorder);
	sim->RunDynamics(data, physics, parameters, experiment);
    return sim->Exit();
}
