#ifndef __GETENERGY_H__
#define __GETENERGY_H__

#include "mainhead.h"

// gets potential energy, calculate kinetic energy and average velocity
void getEnergy(		  DevDataBlock *dev_dat
					, HostDataBlock *host_dat
					,int Ntets
					,int Nnodes
					,float &pE
					,float &kE
					,float &ave_vx
					,float &ave_vy
					,float &ave_vz){


	//need to pitch 1D memory correctly to send to device
	size_t height3 = 3;
	size_t widthNODE = Nnodes;


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

	float peTOTAL = 0.0;
	for(int nt=0;nt<Ntets;nt++){
		peTOTAL+=host_dat->host_pe[nt];
	}//nt

	float tetke,keTOTAL = 0.0,vx,vy,vz;
	ave_vx = 0.0f, ave_vy = 0.0f; ave_vz = 0.0f;
	for(int nt=0;nt<Nnodes;nt++){
		vx = host_dat->host_v[nt+0*Nnodes];
		vy = host_dat->host_v[nt+1*Nnodes];
		vz = host_dat->host_v[nt+2*Nnodes];
		tetke=0.5*host_dat->host_m[nt]*(vx*vx+vy*vy+vz*vz);
		ave_vx += vx;
		ave_vy += vy;
		ave_vz += vz;
		keTOTAL+=tetke;
	}//nt
	float totalVolume = host_dat->host_totalVolume*10.0;
	ave_vx /= float(Nnodes);
	ave_vy /= float(Nnodes);
	ave_vz /= float(Nnodes);
	pE = peTOTAL/totalVolume;
	kE = keTOTAL/totalVolume;

}//getEnergy




#endif //__GETENERGY_H__