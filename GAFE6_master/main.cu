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
#include "mesh.h"
#include "DeviceController.h"

int main()
{

	
	//Get Device properties
	cudaDeviceProp prop;
	HANDLE_ERROR(cudaGetDeviceProperties(&prop,0));
	printf( "Code executing on %s\n\n", prop.name );
	//displayGPUinfo(prop);

	//int Ntets,Nnodes;
	//get dimensions of the mesh
	//get_mesh_dim(Ntets,Nnodes);
	Mesh mesh;
	mesh.loadMeshDim();
	
	//create objects of TetArray and NodeArray class with correct size
	//TetArray Tet = TetArray(Ntets);
	//NodeArray Node = NodeArray(Nnodes);
	mesh.createTetAndNodeArrays();

	//read the mesh into Node and Tet objects
	//get_mesh(Node,Tet,Ntets,Nnodes);
	mesh.loadMesh();

	//get positions of tetrahedra
	//get_tet_pos(Node,Tet,Ntets);
	mesh.calculateTetPositions();

	//set director n for each tetrahedra
	//set_n(Tet,Ntets);
	mesh.loadDirector();

	// comment out GPU calculations while Debugging director sim
	//reorder tetrahedra 
	//gorder_tet(Node,Tet,Ntets);
	//re-order nodes and reassing tetrahedra component lists
	//finish_order(Node,Tet,Ntets,Nnodes);
	mesh.orderTetAndNodeArrays();

	//find initial A's and invert them  store all in Tet object
	//init_As(Node,Tet,Ntets);
	mesh.calculateAMatrices();

	//print spacefilling curve to represent adjacensy between tetrahedra
	//printorder(Tet,Ntets);
	//pritn director
	//Tet.printDirector();
	mesh.printOrderAndDirector();

	//now ready to prepare for dyanmics
	//delcare data stuctures for data on device
	//and host
	//DevDataBlock dev_dat;
	//HostDataBlock host_dat;
	DeviceController gpu_controller(&mesh);

	//Pack data to send to device
	//packdata(Node,Tet,&host_dat,Ntets,Nnodes);
	gpu_controller.packData();

	//send data to device
	//data_to_device(&dev_dat,&host_dat,Ntets,Nnodes);
	gpu_controller.dataHostToDevice();
	gpu_controller.runDynamics();
	////Print Simulation Parameters and Such
	//printf("\n\n Prepared for dynamics with:\n  \
	//			steps/frame	  =	  %d\n    \
	//			Volume        =   %f cm^3\n  \
	//			Mass          =   %f kg\n\n",iterPerFrame,host_dat.host_totalVolume,host_dat.host_totalVolume*materialDensity);
	////=================================================================
	////initillize GPU syncronization arrays
	////will store syncronization information
	////=================================================================
	//int Threads_Per_Block = TPB;
	//int Blocks = (Ntets+Threads_Per_Block)/Threads_Per_Block;
	//int *Syncin,*Syncout,*g_mutex;
	////allocate memory on device for Syncin and Syncoutd
	//
	//HANDLE_ERROR( cudaMalloc( (void**)&Syncin
	//							,Blocks*sizeof(int) ) );
	//HANDLE_ERROR( cudaMalloc( (void**)&Syncout
	//							,Blocks*sizeof(int) ) );
	//int* SyncZeros;
	//SyncZeros = (int*)malloc(Blocks*sizeof(int));
	//for (int i=0;i<Blocks;i++){
	//	SyncZeros[i]=0;
	//}
	//
	//HANDLE_ERROR( cudaMemcpy(Syncin
	//						,SyncZeros
	//						,Blocks*sizeof(int)
	//						,cudaMemcpyHostToDevice ) );
	////allocate global mutex and set =0 
	// HANDLE_ERROR( cudaMalloc( (void**)&g_mutex,
	//                            sizeof(int) ) );
	//HANDLE_ERROR( cudaMemset( g_mutex, 0, sizeof(int) ) );
	// 
	////=================================================================
	////run dynamics
	////=================================================================
	//run_dynamics(&dev_dat,&host_dat,Ntets,Nnodes,Syncin,Syncout,g_mutex);
	////check for CUDA erros
	//any_errors();
	////exit program
	//HANDLE_ERROR( cudaFree( Syncin ) );
	//HANDLE_ERROR(cudaFree( Syncout ) );
	//HANDLE_ERROR(cudaFree( g_mutex ) );
	//exit_program(&dev_dat);
	//*/

    return 0;
}
