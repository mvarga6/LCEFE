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
#include "performance_recorder.h"


//This funciton handles all dynamics which will be run
void run_dynamics(DevDataBlock *dev
				,HostDataBlock *host
				,SimulationParameters *params
				,int *Syncin
				,int *Syncout
				,int *g_mutex
				,std::vector<int>* surf_tets
				,VtkWriter *vtkWriter
				,DataManager *dataManager
				,PerformanceRecorder *recorder)
{
	//======================================
	// Create data procedures needed for 
	// memory operations between GPU and CPU
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
	real pE, kE;
	int Ntets = dev->Ntets;
	int Nnodes = dev->Nnodes;

	// bind data to vtk writer
	//vtkWriter->BindPoints(host->r, Nnodes, DataFormat::LinearizedByDimension, 3);
	//vtkWriter->BindCells(host->TetToNode, Ntets, DataFormat::LinearizedByDimension, CellType::Tetrahedral);

	const real dt = params->Dynamics.Dt;
	const real meshScale = params->Mesh.Scale;
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

	//================================================================
	// initiallize experimental setup contraints
	//================================================================
	

	//================================================================
	// Begin Dynsmics
	//================================================================

	recorder->Create("time-loop")->Start();

	for(int iKern = 0; iKern < nSteps; iKern++)
	{
		//calculate force and send force components to be summed
		force_kernel<<<BlocksTet,Threads_Per_Block>>>(*dev, dt*real(iKern));

		//sum forces and update positions	
		updateKernel<<<BlocksNode,Threads_Per_Block>>>(*dev);

		//sync threads before updating
		cudaThreadSynchronize();
  		recorder->Mark("time-loop");

		//pull data to host then print to files
		if((iKern) % iterPerFrame == 0)
		{
			printf("\n==============================================");
			printf("\nKernel: %d of %d", iKern + 1, nSteps);
			printf("\nTime: %f seconds", real(iKern)*dt);
			recorder->Log("time-loop");
			
			//HANDLE_ERROR(cudaEventRecord(frameStart));
		
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

			fprintf(Eout,"%f %f %f %f\n", real(iKern)*dt, pE, kE, pE+kE);
			fflush(Eout);

		}
		
	}//iKern

	fclose(Eout);

/*	std::string timingFile(params->Output.Base + "_timing.dat");*/

/*	char fout2[128];*/
/*	sprintf(fout2, "%s", timingFile.c_str());*/
/*  	FILE*pout;*/
/*//  	pout = fopen("Performance//timing.dat","w");*/
/*  	pout = fopen(fout2,"w");*/
/*  	fprintf(pout,"nodes = %d\n", dev->Nnodes);*/
/*  	fprintf(pout,"elements = %d\n", dev->Ntets);*/
/*  	fprintf(pout,"forcecalc time (ms) = %f\n", etF / countF);	*/
/*  	fprintf(pout,"update time (ms) = %f\n", etU / countU);*/
/*  	fclose(pout);*/


	//printf("Adev = %f Ahost = %f\n", Acheck[20+(3+2*4)], host->A[20+(3+2*4)]);

	//===================================================================




};

#endif
