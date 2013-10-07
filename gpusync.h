#ifndef __GPUSYNC_H__
#define __GPUSYNC_H__
#include "mainhead.h"
#include "errorhandle.h"
#include <sm_11_atomic_functions.h>

//======================================================
//initialize Ain and Aout on GPU for syncing all threads
//not working right now, jsut doing initialzition in main
//======================================================
/*
void init_sync(int **Syncin,int **Syncout,int Blocks,int **g_mutex){
	
	//allocate memory on device for Syncin and Syncout
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
							

}//init_sync
*/
void end_sync(int **Syncin,int **Syncout,int **g_mutex){

	HANDLE_ERROR( cudaFree( Syncin ) );
	HANDLE_ERROR( cudaFree( Syncout ) );
	HANDLE_ERROR( cudaFree( g_mutex ) );
}


//==========================================
//device code to cary out syncronization 
//this is a variation of the "GPU lock-free"
//syncronization method
//==========================================
__device__ void GPU_sync(int goalVal
					,int *Syncin
					,int *Syncout){


}//GPU_sync


//=========================================
// simple GPU syc
//=========================================
__device__ void simple_GPU_sync(int goalVal,int *g_mutex){
	//thead Id in block
	int tid_in_block = threadIdx.x;

	//use thread 0 for syc
	if (tid_in_block==0){
		atomicAdd(g_mutex, 1);
		while(*g_mutex!=goalVal){
			
		}//g_mutex
	}//tid_in_block
	__syncthreads();


}//simple_GPU_sync





#endif// __GPUSYNC_H__