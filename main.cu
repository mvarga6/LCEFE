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


#include "mainhead.h"
//#include "simulation_parameters.h"
#include "parameters_reader.h"
#include "parameters_writer.h"

#include "constant_cuda_defs.h"

int main(int argc, char *argv[])
{
	// Read simulation parameters
	SimulationParameters parameters;
	ParametersReader *reader = new ParametersReader();
	ParseResult result;
	
	// from cmdline
	result = reader->ReadFromCmdline(argc, argv, parameters);
	
	// from file if given
	if (!parameters.File.empty())
	{
		result = reader->ReadFromFile(parameters.File, parameters);
	}
	
	
	ParametersWriter *writer = new ConsoleWriter();
	writer->Write(parameters);
	
	//return 1;

	//Get commandline arguments
	//parseCommandLine(argc, argv, &parameters);	

	//Get Device properties
	cudaDeviceProp prop;
	HANDLE_ERROR(cudaGetDeviceProperties(&prop,0));
	printf( "Code executing on %s\n\n", prop.name );
	//displayGPUinfo(prop);

	//int Ntets,Nnodes;
	//get dimensions of the mesh
	//get_mesh_dim(Ntets, Nnodes);
	MeshDimensions meshDim = get_gmsh_dim(parameters.Mesh.File);

	//create objects of TetArray and NodeArray class with correct size
	TetArray Tets = TetArray(meshDim.Ntets);
	NodeArray Nodes = NodeArray(meshDim.Nnodes);

	//read the mesh into Node and Tet objects
	//get_mesh(Node,Tet,Ntets,Nnodes);
	get_gmsh(parameters.Mesh.File, Nodes, Tets, parameters.Mesh.Scale);
	
	//const float flatten_Z[3] = {1.0f, 1.0f, 0.75f};
	//Nodes.deform(flatten_Z);
	//Node.eulerRotation(0, PI/2.0, 0);

	//get positions of tetrahedra
	get_tet_pos(Nodes, Tets);

	//set director n for each tetrahedra
	set_n(Tets, &parameters);

	// comment out GPU calculations while Debugging director sim

	//reorder tetrahedra 
	gorder_tet(Nodes, Tets);

	//re-order nodes and reassing tetrahedra component lists
	finish_order(Nodes, Tets);

	//find initial A's and invert them  store all in Tet object
	init_As(Nodes, Tets);

	//print spacefilling curve to represent adjacensy between tetrahedra
	printorder(Tets, parameters.Output.Base);

	//pritn director
	Tets.printDirector(parameters.Output.Base);

	//now ready to prepare for dyanmics
	//delcare data stuctures for data on device
	//and host
	DevDataBlock dev;
	HostDataBlock host;
	std::vector<int> surfTets;

	//Pack data to send to device
	packdata(Nodes, Tets, &host, &surfTets, &parameters);
	
	//send data to device
	data_to_device(&dev, &host, &parameters);

	//Print Simulation Parameters and Such
	printf("\n\nPrepared for dynamics with:\n  \
				steps/frame	  =	  %d\n    \
				Volume        =   %f cm^3\n  \
				Mass          =   %f kg\n\n",
				parameters.Output.FrameRate,
				host.totalVolume,
				host.totalVolume * parameters.Material.Density);


	//=================================================================
	//initillize GPU syncronization arrays
	//will store syncronization information
	//=================================================================
	int Threads_Per_Block = parameters.Gpu.ThreadsPerBlock;
	int Blocks = (Tets.size + Threads_Per_Block) / Threads_Per_Block;
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
	 
	 VtkWriter vtkWriter(parameters.Output.Base);
	 
	//=================================================================
	//run dynamics
	//=================================================================

	run_dynamics(&dev, &host, &parameters, Syncin, Syncout, g_mutex, &surfTets, &vtkWriter);

	//check for CUDA erros
	any_errors();

	//exit program

	HANDLE_ERROR( cudaFree( Syncin ) );
	HANDLE_ERROR(cudaFree( Syncout ) );
	HANDLE_ERROR(cudaFree( g_mutex ) );
	exit_program(&dev);

	//*/

    return 0;
}
