#ifndef __PRINTVTKFRAME_H__
#define __PRINTVTKFRAME_H__

#include "mainhead.h"
#include "illumination_cpu.h"

//=============================================================
//print new vtk file for a time step
void printVTKframe(DevDataBlock *dev
			,HostDataBlock *host
			,std::string outputBase
			,std::vector<int>* illum_list
			,int step){

	//need to pitch 1D memory correctly to send to device
	int Nnodes = dev->Nnodes;
	int Ntets = dev->Ntets;
	//size_t height3 = 3;
	//size_t widthNODE = Nnodes;

/*	HANDLE_ERROR( cudaMemcpy2D(  host->r*/
/*								, widthNODE*sizeof(real)*/
/*								, dev->r*/
/*								, dev->rpitch*/
/*								, widthNODE*sizeof(real)*/
/*								, height3*/
/*								, cudaMemcpyDeviceToHost ) );*/

/*	*/
/*	HANDLE_ERROR( cudaMemcpy2D(  host->v*/
/*								, widthNODE*sizeof(real)*/
/*								, dev->v*/
/*								, dev->vpitch*/
/*								, widthNODE*sizeof(real)*/
/*								, height3*/
/*								, cudaMemcpyDeviceToHost ) );*/

/*	HANDLE_ERROR( cudaMemcpy(  host->pe*/
/*								, dev->pe*/
/*								, Ntets*sizeof(real)*/
/*								, cudaMemcpyDeviceToHost ) );*/

/*	HANDLE_ERROR( cudaMemcpy ( host->S*/
/*								, dev->S*/
/*								, Ntets*sizeof(int)*/
/*								, cudaMemcpyDeviceToHost ) );*/

	//.. optics calculation for setting S
	//int * sloc = new int[Ntets]; // each tet has an S
	//for(int i = 0; i < Ntets; i++)
		//sloc[i] = -S0*SRES; // set all S = 1

	//real light_k[3] = {sin(IANGLE*DEG2RAD), 0, -cos(IANGLE*DEG2RAD)};
	//real light_k[3] = {1, 0, 0};
	
/*	if(step > iterPerFrame*25) {*/
/*		calc_S_from_light(light_k, */
/*				host->r, */
/*				host->TetToNode, */
/*				Ntets, */
/*				Nnodes, */
/*				host->S, */
/*				illum_list,*/
/*				0.2*meshScale, 0.2*meshScale);*/
/*	}*/

/*	//.. copy new S to device*/
/*	HANDLE_ERROR( cudaMemcpy(dev->S, */
/*					host->S, */
/*					Ntets*sizeof(int), */
/*					cudaMemcpyHostToDevice));*/

	std::string meshFile(outputBase + "_mesh");

	char fout[128];
//	sprintf(fout,"VTKOUT//mesh%d.vtk",step);
	sprintf(fout,"%s_%d.vtk", meshFile.c_str(), step);
//	sprintf(fout,VTKNAME,step);
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
	fprintf(out,"SCALARS potentialEnergy real 1\n");
	fprintf(out,"LOOKUP_TABLE default\n");
	real tetpe, peTOTAL = 0.0;
	for(int nt = 0; nt < Ntets; nt++)
	{
		tetpe = host->pe[nt];
		peTOTAL += tetpe;
		//fprintf(out,"%f\n",tetpe+10.0);
		fprintf(out, "%f\n", real(host->S[nt]) / real(SRES)); // print S for debugging
		//fprintf(out, "%f\n", ); // print S for debugging
	}//nt
	//delete [] sloc;

	//peTOTAL = peTOTAL*10000000.0;

	fprintf(out,"\n");
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
		
	/*printf("\nCheck for blowup: %f %f %f\n\n",host_dat->host_r[200+0*Nnodes]
								,host_dat->host_r[200+1*Nnodes]
								,host_dat->host_r[200+2*Nnodes]);*/



}//printVTKframe



#endif //__PRINTVTKFRAME_H__
