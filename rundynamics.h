#ifndef __RUNDYNAMICS_H__
#define __RUNDYNAMICS_H__
#include "mainhead.h"
#include "parameters.h"
#include "printVTKframe.h"
#include "gpuForce.h"
#include "updateKernel.h"
#include "getEnergy.h"

#include "output_writer.h"
#include "data_manager.h"


//This funciton handles all dynamics which will be run
void run_dynamics(DevDataBlock *dev
				,HostDataBlock *host
				,SimulationParameters *params
				,int *Syncin
				,int *Syncout
				,int *g_mutex
				,std::vector<int>* surf_tets
				,VtkWriter *vtkWriter
				,DataManager *dataManager)
{
	//======================================
	// Create data procedures needed
	//======================================
	DataProcedure *getPrintData = new GetPrintData();

	//==============================================================
	//file to write energies to
	//===============================================================

	std::string energyFile(params->Output.Base + "_EvsT.dat");	
	
	char fout[128];
	sprintf(fout, "%s", energyFile.c_str());
	FILE*Eout;
//	Eout = fopen("Output//EvsT.dat","w");
	Eout = fopen(fout,"w");
	float pE, kE;
	int Ntets = dev->Ntets;
	int Nnodes = dev->Nnodes;

	// bind data to vtk writer
	//vtkWriter->BindPoints(host->r, Nnodes, DataFormat::LinearizedByDimension, 3);
	//vtkWriter->BindCells(host->TetToNode, Ntets, DataFormat::LinearizedByDimension, CellType::Tetrahedral);

	const float dt = params->Dynamics.Dt;
	const float meshScale = params->Mesh.Scale;
	const int Threads_Per_Block = params->Gpu.ThreadsPerBlock;
	const int iterPerFrame = params->Output.FrameRate;
	const int nSteps = params->Dynamics.Nsteps;

	//=================================================================
	//claclulate number of blocks to be executed
	//=================================================================
	cudaDeviceProp dev_prop;
	HANDLE_ERROR(cudaGetDeviceProperties(&dev_prop,0));
	int BlocksTet = (Ntets + Threads_Per_Block) / Threads_Per_Block;
	int BlocksNode = (Nnodes + Threads_Per_Block) / Threads_Per_Block;

	printf("execute dynamnics kernel using:\n%d blocks\n%d threads per bock\n",BlocksTet,Threads_Per_Block);

	//size_t widthTETS = Ntets;
	//size_t height16 = 16;

	//float *Acheck;
	float *Acheck = (float*)malloc(Ntets * 16 * sizeof(float));

	//================================================================
	// create start and stop events to measure performance
	//================================================================
	cudaEvent_t startF, stopF, startU, stopU; 
	float elapsedTimeF,elapsedTimeU;
	float etF = 0.0, etU = 0.0;
 	float countF = 0.0, countU = 0.0;

	//================================================================
	// initiallize experimental setup contraints
	//================================================================
	float tableZ = host->min[2] - 0.05*meshScale;	//table position
	float clamps[2] = {host->min[0] + 0.1f,  	//left clamp 
			   host->max[0] - 0.1f}; 	//right clamp 

	//================================================================
	// Begin Dynsmics
	//================================================================

	for(int iKern = 0; iKern < nSteps; iKern++)
	{

  		//timer for force calculation
		HANDLE_ERROR(cudaEventCreate(&startF));
		HANDLE_ERROR(cudaEventCreate(&stopF));
		HANDLE_ERROR(cudaEventRecord(startF,0));

		//calculate force and send force components to be summed
		force_kernel<<<BlocksTet,Threads_Per_Block>>>(*dev, dt*float(iKern));

		//sync threads before updating
		cudaThreadSynchronize();

		//end timer for force kernel
		HANDLE_ERROR(cudaEventRecord(stopF, 0));
		HANDLE_ERROR(cudaEventSynchronize(stopF));
		HANDLE_ERROR(cudaEventElapsedTime(&elapsedTimeF, startF, stopF));
		HANDLE_ERROR(cudaEventDestroy( startF ));
		HANDLE_ERROR(cudaEventDestroy( stopF ));

	  	//start timer for update routine	
	  	HANDLE_ERROR(cudaEventCreate(&startU));
		HANDLE_ERROR(cudaEventCreate(&stopU));
		HANDLE_ERROR(cudaEventRecord(startU,0));

		//sum forces and update positions	
		updateKernel<<<BlocksNode,Threads_Per_Block>>>(*dev, clamps[0], clamps[1], tableZ);

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

		//pull data to host then print to files
		if((iKern) % iterPerFrame == 0)
		{
			printf("\n==============================================");
			printf("\nKernel: %d of %d", iKern + 1, nSteps);
			printf("\nTime: %f seconds", float(iKern)*dt);
			printf("\nSpeed: %f iteartion/s", 1000.0 / (elapsedTimeU + elapsedTimeF));
			
		
			// execute procedure using 
			dataManager->Execute(getPrintData);
		
			//print frame
			printVTKframe(dev
				,host
				,params->Output.Base
				,surf_tets
				,iKern+1);
				
			//vtkWriter->Write(iKern+1);

			


			//print energy
			getEnergy(dev
				,host
				,pE
				,kE);

			fprintf(Eout,"%f %f %f %f\n", float(iKern)*dt, pE, kE, pE+kE);
			fflush(Eout);

		}//if((iKern+1)%iterPerFrame==0)

/*		HANDLE_ERROR(cudaMemcpy2D( Acheck*/
/*				, widthTETS * sizeof(float)*/
/*				, dev->A*/
/*				, dev->Apitch*/
/*				, widthTETS * sizeof(float)*/
/*                , height16*/
/*				, cudaMemcpyDeviceToHost ) );*/

		//reset global mutex
		//HANDLE_ERROR( cudaMemset( g_mutex, 0, sizeof(int) ) );
		
	}//iKern

	fclose(Eout);

	std::string timingFile(params->Output.Base + "_timing.dat");

	char fout2[128];
	sprintf(fout2, "%s", timingFile.c_str());
  	FILE*pout;
//  	pout = fopen("Performance//timing.dat","w");
  	pout = fopen(fout2,"w");
  	fprintf(pout,"nodes = %d\n", dev->Nnodes);
  	fprintf(pout,"elements = %d\n", dev->Ntets);
  	fprintf(pout,"forcecalc time (ms) = %f\n", etF / countF);	
  	fprintf(pout,"update time (ms) = %f\n", etU / countU);
  	fclose(pout);


	//printf("Adev = %f Ahost = %f\n", Acheck[20+(3+2*4)], host->A[20+(3+2*4)]);

	//===================================================================




};

#endif
