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

#include "../include/simulation_parameters.h"
#include "../include/parameters_reader.h"
#include "../include/parameters_writer.h"
#include "../include/output_writer.h"
#include "../include/data_manager.h"
#include "../include/performance_recorder.h"
#include "../include/mesh_operations.h"
#include "../include/director_field.h"
#include "../include/mesh.h"
#include "../include/functions.hpp"
#include "../include/logger.h"
#include "../include/helpers_math.h"
#include "../include/data_procedures.h"
#include "../include/simulation_runner.h"
#include "../include/experiment.h"
#include "../include/pointer.h"
#include "../include/errorhandle.h"

int main(int argc, char *argv[])
{
	// create a console logger
	Logger * log = new ConsoleLogger();

	// for timing data
	PerformanceRecorder * recorder = new PerformanceRecorder(log);

	///
	/// Handle user input
	///

	// create parameters object
	ParametersReader * reader = new ParametersReader(log);
	SimulationParameters * parameters = reader->Read(argc, argv);

	if (!reader->Success())
	{
		printf("%s\n", reader->Result().Message.c_str());
		return (int)reader->Status();
	}
	
	// to write the parameters to console
	ParametersWriter * writer = new LogWriter(log);
	writer->Write(parameters);
	
	///
	/// Create a Mesh from file
	///

	recorder->Create("init")->Start();

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

	mesh->Apply(new CalculateProperties());
	//mesh->Apply(new CalculateVolumes());
	//mesh->Apply(new CalculateAinv());

	///
	/// Endow mesh with liquid crystal properities
	///

	// create director field
	const float3 origin 	 = make_float3(0.0f, 0.0f, 0.0f);
	DirectorField * director = new RadialDirectorField(origin);
	mesh->Apply(new SetDirector(director));
			
	//print director
	//mesh->Tets->printDirector(parameters->Output.Base);

	///
	/// Create data management objects
	///

	// Create Host and Device Data blocks with the mesh
	HostDataBlock 	* host 	= new HostDataBlock(mesh->Nodes, mesh->Tets, mesh->Tris->SelectTag(2), parameters);
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
	//experiment->AddComponent("OrderDynamics", orderDynamics);

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

	// for printing to output files
	VtkWriter * vtkWriter = new VtkWriter(parameters->Output.Base);

	// Create the simulation running environment
	SimulationRunner * sim = new SimulationRunner(vtkWriter, log, recorder);
	sim->RunDynamics(data, physics, parameters, experiment);
    return sim->Exit();
}
