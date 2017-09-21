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


int main(int argc, char *argv[])
{
	//Get commandline arguments
	parseCommandLine(argc, argv);

	//Get Device properties
	cudaDeviceProp prop;
	HANDLE_ERROR(cudaGetDeviceProperties(&prop,0));
	printf( "Code executing on %s\n\n", prop.name );
	//displayGPUinfo(prop);

	int Ntets,Nnodes;
	//get dimensions of the mesh
	//get_mesh_dim(Ntets, Nnodes);
	get_gmsh_dim(std::string(MESHFILE), Ntets, Nnodes);

	//create objects of TetArray and NodeArray class with correct size
	TetArray Tet = TetArray(Ntets);
	NodeArray Node = NodeArray(Nnodes);

	//read the mesh into Node and Tet objects
	//get_mesh(Node,Tet,Ntets,Nnodes);
	get_gmsh(std::string(MESHFILE), Node, Tet, Ntets, Nnodes);
	
	const float flatten_Z[3] = {1.0f, 1.0f, 0.75f};
	Node.deform(flatten_Z);
	//Node.eulerRotation(0, PI/2.0, 0);

	//get positions of tetrahedra
	get_tet_pos(Node, Tet, Ntets);

	//set director n for each tetrahedra
	set_n(Tet, Ntets);

	// comment out GPU calculations while Debugging director sim

	//reorder tetrahedra 
	gorder_tet(Node, Tet, Ntets);

	//re-order nodes and reassing tetrahedra component lists
	finish_order(Node, Tet, Ntets, Nnodes);

	//find initial A's and invert them  store all in Tet object
	init_As(Node, Tet, Ntets);

	//print spacefilling curve to represent adjacensy between tetrahedra
	printorder(Tet, Ntets);

	//pritn director
	Tet.printDirector();

	//now ready to prepare for dyanmics
	//delcare data stuctures for data on device
	//and host
	DevDataBlock dev;
	HostDataBlock host;
	std::vector<int> surfTets;

	//Pack data to send to device
	packdata(Node, Tet, &host, Ntets, Nnodes, &surfTets);

	//send data to device
	data_to_device(&dev, &host);


	//Print Simulation Parameters and Such
	printf("\n\n Prepared for dynamics with:\n  \
				steps/frame	  =	  %d\n    \
				Volume        =   %f cm^3\n  \
				Mass          =   %f kg\n\n",
				iterPerFrame,
				host.totalVolume,
				host.totalVolume * materialDensity);




	//=================================================================
	//initillize GPU syncronization arrays
	//will store syncronization information
	//=================================================================
	int Threads_Per_Block = TPB;
	int Blocks = (Ntets+Threads_Per_Block)/Threads_Per_Block;
	int *Syncin,*Syncout,*g_mutex;
	//allocate memory on device for Syncin and Syncoutd

	
	HANDLE_ERROR( cudaMalloc( (void**)&Syncin
								,Blocks*sizeof(int) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&Syncout
								,Blocks*sizeof(int) ) );

	int* SyncZeros;
	SyncZeros = (int*)malloc(Blocks*sizeof(int));
	for (int i=0;i<Blocks;i++){
		SyncZeros[i]=0;
	}
	
	HANDLE_ERROR( cudaMemcpy(Syncin
							,SyncZeros
							,Blocks*sizeof(int)
							,cudaMemcpyHostToDevice ) );
	//allocate global mutex and set =0 
	 HANDLE_ERROR( cudaMalloc( (void**)&g_mutex,
                              sizeof(int) ) );
     HANDLE_ERROR( cudaMemset( g_mutex, 0, sizeof(int) ) );
	 
	//=================================================================
	//run dynamics
	//=================================================================

	run_dynamics(&dev, &host, Syncin, Syncout, g_mutex, &surfTets);

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
