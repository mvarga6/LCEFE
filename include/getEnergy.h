#ifndef __GETENERGY_H__
#define __GETENERGY_H__

#include "mainhead.h"


void getEnergy(DevDataBlock *dev, HostDataBlock *host, real &pE, real &kE){


	//need to pitch 1D memory correctly to send to device
	int Nnodes = dev->Nnodes;
	int Ntets = dev->Ntets;
	//size_t height3 = 3;
	//size_t widthNODE = Nnodes;

/*	HANDLE_ERROR( cudaMemcpy2D(  host->v*/
/*								, widthNODE*sizeof(real)*/
/*								, dev->v*/
/*								, dev->vpitch*/
/*								, widthNODE*sizeof(real)*/
/*								, height3*/
/*								, cudaMemcpyDeviceToHost ) );*/

/*	HANDLE_ERROR( cudaMemcpy(  host->pe*/
/*								, dev->pe*/
/*								, Ntets * sizeof(real)*/
/*								, cudaMemcpyDeviceToHost ) );*/

	real peTOTAL = 0.0;
	for(int nt = 0; nt < Ntets; nt++)
	{
		peTOTAL += host->pe[nt];
	}//nt

	real tetke, keTOTAL = 0.0, vx, vy, vz;
	for(int nt = 0; nt < Nnodes; nt++){
		vx = host->v[nt+0*Nnodes];
		vy = host->v[nt+1*Nnodes];
		vz = host->v[nt+2*Nnodes];
		tetke = 0.5 * host->m[nt] * (vx*vx + vy*vy + vz*vz);
		keTOTAL+=tetke;
	}//nt
	real totalVolume = host->totalVolume * 10.0;

	pE = peTOTAL / totalVolume;
	kE = keTOTAL / totalVolume;

}//getEnergy




#endif //__GETENERGY_H__
