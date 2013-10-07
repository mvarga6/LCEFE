#ifndef __PRINTVTKFRAME_H__
#define __PRINTVTKFRAME_H__

#include "mainhead.h"

//=============================================================
//print new vtk file for a time step
void printVTKframe(   DevDataBlock *dev_dat
					, HostDataBlock *host_dat
					,int Ntets
					,int Nnodes
					,int step){

	//need to pitch 1D memory correctly to send to device
	size_t height3 = 3;
	size_t widthNODE = Nnodes;


	HANDLE_ERROR( cudaMemcpy2D(  host_dat->host_r
								, widthNODE*sizeof(float)
								, dev_dat->dev_r
								, dev_dat->dev_rpitch
								, widthNODE*sizeof(float)
								, height3
								, cudaMemcpyDeviceToHost ) );

	
	HANDLE_ERROR( cudaMemcpy2D(  host_dat->host_v
								, widthNODE*sizeof(float)
								, dev_dat->dev_v
								, dev_dat->dev_vpitch
								, widthNODE*sizeof(float)
								, height3
								, cudaMemcpyDeviceToHost ) );

	HANDLE_ERROR( cudaMemcpy(  host_dat->host_pe
								, dev_dat->dev_pe
								, Ntets*sizeof(float)
								, cudaMemcpyDeviceToHost ) );
								

	char fout[60];
	sprintf(fout,"VTKOUT//mesh%d.vtk",step);
	FILE*out;
	out = fopen(fout,"w");

	//header
	fprintf(out,"# vtk DataFile Version 3.1\n");
	fprintf(out,"Tetrahedral Mesh Visualization\n");
	fprintf(out,"ASCII\n");
	fprintf(out,"DATASET UNSTRUCTURED_GRID\n");
	fprintf(out,"\n");

	fprintf(out,"POINTS %d FLOAT\n",Nnodes);


	//write node points
	for(int n=0;n<Nnodes;n++){
		fprintf(out,"%f %f %f\n",host_dat->host_r[n+0*Nnodes]
								,host_dat->host_r[n+1*Nnodes]
								,host_dat->host_r[n+2*Nnodes]);
	}//n
	fprintf(out,"\n");

	//print cells
	fprintf(out,"CELLS %d %d\n",Ntets,5*Ntets);
	for(int nt=0;nt<Ntets;nt++){
		fprintf(out,"4 %d %d %d %d\n",host_dat->host_TetToNode[nt+Ntets*0]
									 ,host_dat->host_TetToNode[nt+Ntets*1]
									 ,host_dat->host_TetToNode[nt+Ntets*2]
									 ,host_dat->host_TetToNode[nt+Ntets*3]);
	}//nt
	fprintf(out,"\n");


	fprintf(out,"CELL_TYPES %d\n",Ntets);
	for(int nt=0;nt<Ntets;nt++){
		fprintf(out,"10\n");
	}
	fprintf(out,"\n");

	fprintf(out,"CELL_DATA %d\n",Ntets);
	fprintf(out,"SCALARS potentialEnergy FLOAT 1\n");
	fprintf(out,"LOOKUP_TABLE default\n");
	float tetpe,peTOTAL = 0.0;
	for(int nt=0;nt<Ntets;nt++){
		tetpe=host_dat->host_pe[nt];
		peTOTAL+=tetpe;
		fprintf(out,"%f\n",tetpe+10.0);
	}//nt

	//peTOTAL = peTOTAL*10000000.0;



	fprintf(out,"\n");

	fclose(out);		//close output file

	float tetke,keTOTAL = 0.0,vx,vy,vz;
	for(int nt=0;nt<Nnodes;nt++){
		vx = host_dat->host_v[nt+0*Nnodes];
		vy = host_dat->host_v[nt+1*Nnodes];
		vz = host_dat->host_v[nt+2*Nnodes];
		tetke=0.5*host_dat->host_m[nt]*(vx*vx+vy*vy+vz*vz);
		keTOTAL+=tetke;
	}//nt
	keTOTAL = keTOTAL;
	float totalVolume = host_dat->host_totalVolume*10.0;
	printf("pE = %f J/cm^3  kE = %f J/cm^3  pE+kE = %f J/cm^3\n",peTOTAL/totalVolume,keTOTAL/totalVolume,(peTOTAL+keTOTAL)/totalVolume);
	/*printf("\nCheck for blowup: %f %f %f\n\n",host_dat->host_r[200+0*Nnodes]
								,host_dat->host_r[200+1*Nnodes]
								,host_dat->host_r[200+2*Nnodes]);*/



}//printVTKframe



#endif //__PRINTVTKFRAME_H__