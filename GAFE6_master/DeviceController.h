#pragma once

#include "mainhead.h"
#include "mesh.h"

class DeviceController {
	cudaDeviceProp * props;
	DevDataBlock * dev_dat;
	HostDataBlock * host_dat;
	Mesh * mesh;

	int * sync_in;
	int * sync_out;
	int * g_mutex;
	
	int threads_per_block;
	int tet_blocks;
	int node_blocks;

public:
	DeviceController(Mesh *msh, int threadsPerBlock = TPB) : mesh(msh), props(new cudaDeviceProp){
		HANDLE_ERROR(cudaGetDeviceProperties(this->props, 0));
		displayGPUinfo(props);
		this->dev_dat = new DevDataBlock;
		this->host_dat = new HostDataBlock;
		this->threads_per_block = threadsPerBlock;
		this->tet_blocks = (this->mesh->Ntets / threadsPerBlock) + 1;
		this->node_blocks = (this->mesh->Nnodes / threadsPerBlock) + 1;
	}
	// ----------------------------------------------------------------
	~DeviceController(){
		//exit program
		HANDLE_ERROR(cudaFree(this->sync_in));
		HANDLE_ERROR(cudaFree(this->sync_out));
		HANDLE_ERROR(cudaFree(this->g_mutex));
		exit_program(this->dev_dat);
	}
	// ----------------------------------------------------------------
	inline void packData(){
		packdata(this->mesh->nodeArray,
			this->mesh->tetArray,
			this->host_dat,
			this->mesh->Ntets,
			this->mesh->Nnodes);
	}
	// ----------------------------------------------------------------
	inline void dataHostToDevice(){
		data_to_device(this->dev_dat,
			this->host_dat,
			this->mesh->Ntets,
			this->mesh->Nnodes);
		HANDLE_ERROR(cudaMalloc((void**)&this->sync_in, this->tet_blocks*sizeof(int)));
		HANDLE_ERROR(cudaMalloc((void**)&this->sync_out, this->tet_blocks*sizeof(int)));
		HANDLE_ERROR(cudaMemset(this->sync_in, 0, this->tet_blocks*sizeof(int)));
		HANDLE_ERROR(cudaMalloc((void**)&this->g_mutex, sizeof(int)));
		HANDLE_ERROR(cudaMemset(this->g_mutex, 0, sizeof(int)));

	}
	// ----------------------------------------------------------------
	inline void runDynamics(){
		//=================================================================
		//initillize GPU syncronization arrays
		//will store syncronization information
		//=================================================================
		//
		//int Threads_Per_Block = TPB;
		//int Blocks = (this->mesh->Ntets+Threads_Per_Block)/Threads_Per_Block;
		//
		//int *Syncin,*Syncout,*g_mutex;
		//allocate memory on device for Syncin and Syncoutd
		//HANDLE_ERROR(cudaMalloc((void**)&Syncin, Blocks*sizeof(int)));
		//HANDLE_ERROR(cudaMalloc((void**)&Syncout, Blocks*sizeof(int)));
		//
		//int* SyncZeros;
		//SyncZeros = (int*)malloc(Blocks*sizeof(int));
		//for (int i=0;i<Blocks;i++){
		//	SyncZeros[i]=0;
		//}
		//
		//HANDLE_ERROR( cudaMemcpy(Syncin
		//					,SyncZeros
		//					,Blocks*sizeof(int)
		//					,cudaMemcpyHostToDevice ) );
		//
		//allocate global mutex and set =0 
		//HANDLE_ERROR( cudaMalloc( (void**)&g_mutex, sizeof(int) ) );
		//HANDLE_ERROR( cudaMemset( g_mutex, 0, sizeof(int) ) );
		//=================================================================
		//run dynamics
		//=================================================================
		run_dynamics(this->dev_dat,
					 this->host_dat,
					 this->mesh->Ntets,
					 this->mesh->Nnodes,
					 this->sync_in,
					 this->sync_out,
					 this->g_mutex,
					 this->threads_per_block,
					 this->tet_blocks,
					 this->node_blocks);

		//check for CUDA erros
		any_errors();
		////exit program
		//HANDLE_ERROR( cudaFree( Syncin ) );
		//HANDLE_ERROR(cudaFree( Syncout ) );
		//HANDLE_ERROR(cudaFree( g_mutex ) );
		//exit_program(this->dev_dat);
	}
};