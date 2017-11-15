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
//#include "constant_cuda_defs.h"
#include "performance_recorder.h"
#include "mesh_operations.h"
#include "director_field.h"
#include "mesh.h"
#include "functions.hpp"
#include "logger.h"
#include "helpers_math.h"

// these will go away into their own service class
#include "getAs.h"
#include "setn.h"
#include "printmeshorder.h"
#include "packdata.h"
#include "errorhandle.h"
#include "datatodevice.h"
#include "anyerrors.h"
#include "exit_program.h"

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
		// TODO: Index assignment should happen when reading mesh automatically
		//MeshOperation * reIndex = ;
		mesh->Apply(new ReassignIndices());
		
		log->Msg("Mesh Loaded from cache!");
	}
	
	const float3 origin = make_float3(0.0f, 0.0f, 0.0f);
	DirectorField * director = new RadialDirectorField(origin);
	
	mesh->Apply(new CalculateVolumes());
	mesh->Apply(new CalculateAinv());
	mesh->Apply(new SetDirector(director));
			
	
	//pritn director
	mesh->Tets->printDirector(parameters.Output.Base);

	//now ready to prepare for dyanmics
	//delcare data stuctures for data on device
	//and host
	DevDataBlock dev;
	HostDataBlock host(mesh->Nodes, mesh->Tets, &parameters);
	DataManager * dataManager = new DataManager(&host, &dev);
	
	std::vector<int> surfTets;

	//Pack data to send to device
	//packdata(*mesh->Nodes, *mesh->Tets, &host, &surfTets, &parameters);
	
	//send data to device
	data_to_device(&dev, &host, &parameters, dataManager);

	//Print Simulation Parameters and Such
	printf("\n\nPrepared for dynamics with:\nsteps/frame: %d\nVolume: %f cm^3\nMass: %f kg\n",
				parameters.Output.FrameRate,
				host.totalVolume,
				host.totalVolume * parameters.Material.Density);


	//=================================================================
	//initillize GPU syncronization arrays
	//will store syncronization information
	//=================================================================
	int Threads_Per_Block = parameters.Gpu.ThreadsPerBlock;
	int Blocks = (mesh->Tets->size + Threads_Per_Block) / Threads_Per_Block;
	int *Syncin,*Syncout,*g_mutex, *SyncZeros;
	//allocate memory on device for Syncin and Syncoutd
	
	HANDLE_ERROR( cudaMalloc( (void**)&Syncin, Blocks*sizeof(int) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&Syncout, Blocks*sizeof(int) ) );
	SyncZeros = (int*)malloc(Blocks*sizeof(int));
	
	for (int i = 0; i < Blocks; i++)
	{
		SyncZeros[i]=0;
	}
	
	HANDLE_ERROR( cudaMemcpy(Syncin, SyncZeros, Blocks*sizeof(int), cudaMemcpyHostToDevice ) );
	//allocate global mutex and set =0 
	HANDLE_ERROR( cudaMalloc( (void**)&g_mutex, sizeof(int) ) );
	HANDLE_ERROR( cudaMemset( g_mutex, 0, sizeof(int) ) );
	 
	// Move this somewhere else
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
	run_dynamics(&dev, &host, &parameters, Syncin, Syncout, g_mutex, &surfTets, vtkWriter, dataManager, recorder);

	//check for CUDA erros
	any_errors();

	//exit program

	HANDLE_ERROR(cudaFree( Syncin ) );
	HANDLE_ERROR(cudaFree( Syncout ) );
	HANDLE_ERROR(cudaFree( g_mutex ) );
	exit_program(&dev);

	//*/

    return 0;
}
