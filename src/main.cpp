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
#include "constant_cuda_defs.h"
#include "performance_recorder.h"
#include "mesh_optimizer.h"
#include "director_field.h"
#include "mesh.h"
#include "functions.hpp"

// these will go away into their own service class
#include "getAs.h"
#include "printmeshorder.h"
#include "packdata.h"
#include "errorhandle.h"
#include "datatodevice.h"
#include "anyerrors.h"
#include "exit_program.h"

int main(int argc, char *argv[])
{
	// Move this somewhere else
	//Get Device properties
	cudaDeviceProp prop;
	HANDLE_ERROR(cudaGetDeviceProperties(&prop,0));
	printf( "Code executing on %s\n\n", prop.name );
	//displayGPUinfo(prop);

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
		
	// for timing data
	PerformanceRecorder * recorder = new PerformanceRecorder();;
	recorder->Create("init")->Start();
	
	// to write the parameters to console
	ParametersWriter * writer = new ConsoleWriter();
	writer->Write(parameters);
	
	// for printing to output files
	VtkWriter * vtkWriter = new VtkWriter(parameters.Output.Base);
	
	// the mesh object
	Mesh * mesh = new Mesh(&parameters);
	
	//int Ntets,Nnodes;
	//get dimensions of the mesh
	//get_mesh_dim(Ntets, Nnodes);
	//MeshDimensions meshDim = get_gmsh_dim(parameters.Mesh.File);
	if (!mesh->Load())
	{
		// log failure
		exit(10);
	}

	//create objects of TetArray and NodeArray class with correct size
	//TetArray Tets = TetArray(meshDim.Ntets);
	//NodeArray Nodes = NodeArray(meshDim.Nnodes);

	//read the mesh into Node and Tet objects
	//get_mesh(Node,Tet,Ntets,Nnodes);
	//get_gmsh(parameters.Mesh.File, Nodes, Tets, parameters.Mesh.Scale);
	
	//const float flatten_Z[3] = {1.0f, 1.0f, 0.75f};
	//Nodes.deform(flatten_Z);
	//Node.eulerRotation(0, PI/2.0, 0);

	//get positions of tetrahedra
	//get_tet_pos(Nodes, Tets);

	//set director n for each tetrahedra
	//set_n(Tets, &parameters);
	
	// Get the director field (default for now)
	DirectorField * director = new UniformField(0.0f, 0.0f);
	
	//UnivariableFunction *theta_of_x = new Linear({1.0f});
	//UnivariableFunction *phi_of_y = new Sinusoinal(/* some simulation length */);	
	//ScalerField3D * theta = new MultiplicativeField3D(theta_of_x);
	//ScalerField3D * phi = new AdditiveField3D(NULL, phi_of_y);	
	//DirectorField * director = new CartesianDirectorField(theta, phi);
	
	mesh->SetDirector(director);


	// comment out GPU calculations while Debugging director sim

	//reorder tetrahedra 
	//gorder_tet(Nodes, Tets);

	//re-order nodes and reassing tetrahedra component lists
	//finish_order(Nodes, Tets);
	
	// optimize the mesh
	MeshOptimizer * simpleSort = new SortOnTetrahedraPosition();
	MeshOptimizer * mcReorder = new MonteCarloMinimizeDistanceBetweenPairs(300, 0.01f, 0.999999f);
	MeshOptimizer * reIndex = new ReassignIndices();
	
	mesh->Apply(simpleSort);
	mesh->Apply(mcReorder);
	mesh->Apply(reIndex);

	//find initial A's and invert them  store all in Tet object
	init_As(*mesh->Nodes, *mesh->Tets);

	//print spacefilling curve to represent adjacensy between tetrahedra
	printorder(*mesh->Tets, parameters.Output.Base);

	//pritn director
	mesh->Tets->printDirector(parameters.Output.Base);

	//now ready to prepare for dyanmics
	//delcare data stuctures for data on device
	//and host
	DevDataBlock dev;
	HostDataBlock host;
	DataManager * dataManager = new DataManager(&host, &dev);
	
	std::vector<int> surfTets;

	//Pack data to send to device
	packdata(*mesh->Nodes, *mesh->Tets, &host, &surfTets, &parameters);
	
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
