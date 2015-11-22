#ifndef __RUNDYNAMICS_H__
#define __RUNDYNAMICS_H__
#include "mainhead.h"
#include "parameters.h"
#include "printVTKframe.h"
#include "gpuForce.h"
#include "updateKernel.h"
#include "getEnergy.h"



//This funciton handles all dynamics which will be run
void run_dynamics(DevDataBlock *data
					, HostDataBlock *host_data
					, int Ntets
					, int Nnodes
					, int *Syncin
					, int *Syncout
					, int *g_mutex
					, int Threads_Per_Block
					, int BlocksTet
					, int BlocksNode){

	//Print Simulation Parameters and Such
	printf("\n\n Prepared for dynamics with:\n  \
		   		steps/frame	  =	  %d\n    \
				Volume        =   %f cm^3\n  \
				Mass          =   %f kg\n\n",
				iterPerFrame,
				host_data->host_totalVolume,
				host_data->host_totalVolume*materialDensity);
	//==============================================================
	//file to write energies to
	//===============================================================
	FILE * Eout;
	Eout = fopen("Output//EvsT.dat","w");
	float pE,kE;
	//=================================================================
	//claclulate number of blocks to be executed
	//=================================================================
	//cudaDeviceProp dev_prop;
	//HANDLE_ERROR(cudaGetDeviceProperties(&dev_prop,0));
	//int Threads_Per_Block = TPB;
	//int BlocksTet = (Ntets+Threads_Per_Block)/Threads_Per_Block;
	//int BlocksNode = (Nnodes+Threads_Per_Block)/Threads_Per_Block;

	printf("execute dynamnics kernel using:\n%d blocks\n%d threads per bock\n",BlocksTet,Threads_Per_Block);
	
	size_t widthTETS = Ntets;
	size_t height16 = 16;
	//float *Acheck;
	//Acheck = (float*)malloc(Ntets*16*sizeof(float));

	//================================================================
	// create start and stop events to measure performance
	//================================================================
	cudaEvent_t startF, stopF, startU, stopU; 
	float elapsedTimeF,elapsedTimeU;
	float etF = 0.0, etU = 0.0;
	float countF = 0.0, countU = 0.0;
	//================================================================
	// Begin Dynamics
	//================================================================
	for(int iKern = 0; iKern < NSTEPS; iKern++){
	
		//timer for force calculation
		HANDLE_ERROR(cudaEventCreate(&startF));
		HANDLE_ERROR(cudaEventCreate(&stopF));
		HANDLE_ERROR(cudaEventRecord(startF,0));

		//calculate force and send force components to be summed
		force_kernel<<< BlocksTet,Threads_Per_Block >>>( data->dev_A
											  , data->dev_Apitch
											  , data->dev_dF
											  , data->dev_dFpitch
											  , data->dev_TetNodeRank
											  , Ntets 
											  , data->dev_v
											  , data->dev_vpitch
											  , data->dev_pe
											  , data->dev_TetVol
											  , data->dev_ThPhi
											  , data->dev_TetToNode
											  , data->dev_TetToNodepitch
											  , dt*float(iKern));

		//sync threads before updating
		cudaThreadSynchronize();

		//end timer for force kernel
		HANDLE_ERROR(cudaEventRecord(stopF, 0));
		HANDLE_ERROR(cudaEventSynchronize(stopF));
		HANDLE_ERROR(cudaEventElapsedTime(&elapsedTimeF, startF, stopF));
		HANDLE_ERROR( cudaEventDestroy( startF ));
		HANDLE_ERROR( cudaEventDestroy( stopF ));

		//start timer for update routine	
		HANDLE_ERROR(cudaEventCreate(&startU));
		HANDLE_ERROR(cudaEventCreate(&stopU));
		HANDLE_ERROR(cudaEventRecord(startU,0));

		//sum forces and update positions
		updateKernel<<< BlocksNode,Threads_Per_Block >>>( data->dev_dF
												, data->dev_dFpitch
												, data->dev_F
												, data->dev_Fpitch
												, Nnodes
												, data->dev_nodeRank
												, data->dev_v
												, data->dev_vpitch
												, data->dev_r
												, data->dev_rpitch
												, data->dev_m);

		//sync threads before updating
		cudaThreadSynchronize();

		//end timer for force kernel
		HANDLE_ERROR(cudaEventRecord(stopU, 0));
		HANDLE_ERROR(cudaEventSynchronize(stopU));
		HANDLE_ERROR(cudaEventElapsedTime(&elapsedTimeU, startU, stopU));
		HANDLE_ERROR( cudaEventDestroy( startU ));
		HANDLE_ERROR( cudaEventDestroy( stopU ));

		//update timer data
		etF += elapsedTimeF;
		etU += elapsedTimeU;
		countF += 1.0;
		countU += 1.0;

		if((iKern) % iterPerFrame == 0){
		
			//print calculation speed
			printf("\nIteration rate:  %f  iteartion/s \n kernel %d of %d\n"
								,1000.0/(elapsedTimeU+elapsedTimeF)
								,iKern+1
								,NSTEPS);

			//print frame
			printVTKframe(   data
						,host_data
						,Ntets
						,Nnodes
						,iKern+1);
			printf("time = %f seconds\n", float(iKern)*dt);

			//print energy
			getEnergy(	 data
					,host_data
					,Ntets
					,Nnodes
					,pE
					,kE );
			fprintf(Eout,"%f %f %f %f\n",float(iKern)*dt,pE,kE,pE+kE);
			fflush(Eout);

		}//if((iKern+1)%iterPerFrame==0)

		/*HANDLE_ERROR(cudaMemcpy2D(Acheck
								, widthTETS*sizeof(float)
								, data->dev_A
								, data->dev_Apitch
								, widthTETS*sizeof(float)
                                , height16
								, cudaMemcpyDeviceToHost ) );*/
	

		//reset global mutex
		HANDLE_ERROR(cudaMemset( g_mutex, 0, sizeof(int)));
	}//iKern

	fclose(Eout);

	FILE*pout;
	pout = fopen("Performance//timing.dat","w");
	fprintf(pout,"nodes = %d\n",Nnodes);
	fprintf(pout,"elements = %d\n",Ntets);
	fprintf(pout,"forcecalc time (ms) = %f\n",etF/countF);	
	fprintf(pout,"update time (ms) = %f\n",etU/countU);
	fclose(pout);	

	//printf("Adev = %f Ahost = %f\n",Acheck[20+(3+2*4)],host_data->host_A[20+(3+2*4)]);
	//===================================================================
};

#endif
