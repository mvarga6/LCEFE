#pragma once

#include "mainhead.h"
#include "mesh.h"

class DeviceController {
	cudaDeviceProp * props;
	DevDataBlock * dev_dat;
	HostDataBlock * host_dat;
	Mesh * mesh;
public:
	DeviceController(Mesh *msh) : mesh(msh), props(new cudaDeviceProp){
		HANDLE_ERROR(cudaGetDeviceProperties(this->props, 0));
		displayGPUinfo(props);
		this->dev_dat = new DevDataBlock;
		this->host_dat = new HostDataBlock;
	}
	~DeviceController(){};
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
	}
	// ----------------------------------------------------------------
	inline void runDynamics(){
		//Print Simulation Parameters and Such
		printf("\n\n Prepared for dynamics with:\n  \
				steps/frame	  =	  %d\n    \
				Volume        =   %f cm^3\n  \
				Mass          =   %f kg\n\n",
				iterPerFrame,
				host_dat->host_totalVolume,
				host_dat->host_totalVolume*materialDensity);
		//=================================================================
		//initillize GPU syncronization arrays
		//will store syncronization information
		//=================================================================

		int Threads_Per_Block = TPB;
		int Blocks = (this->mesh->Ntets+Threads_Per_Block)/Threads_Per_Block;

		int *Syncin,*Syncout,*g_mutex;
		//allocate memory on device for Syncin and Syncoutd
		HANDLE_ERROR(cudaMalloc((void**)&Syncin, Blocks*sizeof(int)));
		HANDLE_ERROR(cudaMalloc((void**)&Syncout, Blocks*sizeof(int)));

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
		HANDLE_ERROR( cudaMalloc( (void**)&g_mutex, sizeof(int) ) );
		HANDLE_ERROR( cudaMemset( g_mutex, 0, sizeof(int) ) );
	 
		//=================================================================
		//run dynamics
		//=================================================================
		run_dynamics(this->dev_dat,this->host_dat,this->mesh->Ntets,this->mesh->Nnodes,Syncin,Syncout,g_mutex);

		//check for CUDA erros
		any_errors();

		//exit program
		HANDLE_ERROR( cudaFree( Syncin ) );
		HANDLE_ERROR(cudaFree( Syncout ) );
		HANDLE_ERROR(cudaFree( g_mutex ) );
		exit_program(this->dev_dat);
	}
};