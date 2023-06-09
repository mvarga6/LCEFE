#ifndef __PRINTVTKFRAME_H__
#define __PRINTVTKFRAME_H__

//#include "mainhead.h"
#include <stdio.h>
#include "illumination_cpu.h"

//=============================================================
//print new vtk file for a time step
void printVTKframe(DevDataBlock *dev
			,HostDataBlock *host
			,std::string outputBase
			,int step){

	//need to pitch 1D memory correctly to send to device
	int Nnodes = dev->Nnodes;
	int Ntets = dev->Ntets;


	std::string meshFile(outputBase + "_mesh");

	char fout[128];
	sprintf(fout,"%s_%d.vtk", meshFile.c_str(), step);
	FILE*out;
	out = fopen(fout,"w");

	//header
	fprintf(out,"# vtk DataFile Version 3.1\n");
	fprintf(out,"Tetrahedral Mesh Visualization\n");
	fprintf(out,"ASCII\n");
	fprintf(out,"DATASET UNSTRUCTURED_GRID\n");
	fprintf(out,"\n");

	fprintf(out,"POINTS %d float\n",Nnodes);


	//write node points
	for(int n = 0; n < Nnodes; n++)
	{
		fprintf(out,"%f %f %f\n",host->r[n+0*Nnodes]
					,host->r[n+1*Nnodes]
					,host->r[n+2*Nnodes]);
	}//n
	fprintf(out,"\n");

	//print cells
	fprintf(out,"CELLS %d %d\n", Ntets, 5*Ntets);
	for(int nt = 0; nt < Ntets; nt++)
	{
		fprintf(out,"4 %d %d %d %d\n",host->TetToNode[nt+Ntets*0]
						,host->TetToNode[nt+Ntets*1]
						,host->TetToNode[nt+Ntets*2]
						,host->TetToNode[nt+Ntets*3]);
	}//nt
	fprintf(out,"\n");


	fprintf(out,"CELL_TYPES %d\n", Ntets);
	for(int nt = 0; nt < Ntets; nt++)
	{
		fprintf(out,"10\n");
	}
	fprintf(out,"\n");

	fprintf(out,"CELL_DATA %d\n", Ntets);
	fprintf(out,"SCALARS PotentialEnergy float 1\n");
	fprintf(out,"LOOKUP_TABLE default\n");
	real tetpe, peTOTAL = 0.0;
	for(int nt = 0; nt < Ntets; nt++)
	{
		tetpe = host->pe[nt];
		peTOTAL += tetpe;
		fprintf(out,"%f\n", tetpe);
	}//nt
	//delete [] sloc;


	fprintf(out,"\n");
	//fprintf(out,"CELL_DATA %d\n", Ntets);
	//fprintf(out,"LOOKUP_TABLE default\n");
	fprintf(out,"VECTORS Director float\n");
	real thphi, nTh,nPhi, theta, phi, nx, ny, nz;
	for(int t = 0; t < Ntets; t++)
	{
		thphi = real(host->ThPhi[t]);

		nTh = floor(thphi/10000.0);
		nPhi = thphi-nTh*10000.0;

		theta = nTh*PI/1000.0;
		phi = nPhi*PI/500.0;

		nx = sin(theta)*cos(phi);
		ny = sin(theta)*sin(phi);
		nz = cos(theta);

		fprintf(out,"%f %f %f\n", nx, ny, nz);
	}//nt


	////Scalar Order Parameter
	//////////////////////////
	fprintf(out,"SCALARS ScalarOrderParameter float 1\n");
	fprintf(out,"LOOKUP_TABLE default\n");
	real S;
	for(int t = 0; t < Ntets; t++)
	{
		S = host->S[t] ; //plus 1 for actual experimental S


		fprintf(out,"%f\n", S);
	}//nt
	//////////////////////////////
	////END Scalar Order Parameter

	fclose(out);		//close output file

	real tetke, keTOTAL = 0.0, vx, vy, vz;
	for(int nt = 0; nt < Nnodes; nt++)
	{
		vx = host->v[nt+0*Nnodes];
		vy = host->v[nt+1*Nnodes];
		vz = host->v[nt+2*Nnodes];
		tetke = 0.5 * host->m[nt] * (vx*vx + vy*vy + vz*vz);
		keTOTAL += tetke;
	}//nt

	keTOTAL = keTOTAL;
	real totalVolume = host->totalVolume*10.0;
	printf("\npE = %f J/cm^3\nkE = %f J/cm^3\npE+kE = %f J/cm^3\n",
		peTOTAL / totalVolume,
		keTOTAL / totalVolume,
		(peTOTAL + keTOTAL) / totalVolume);




}//printVTKframe



#endif //__PRINTVTKFRAME_H__
