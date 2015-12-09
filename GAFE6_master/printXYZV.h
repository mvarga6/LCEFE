#pragma once

#include <fstream>
#include "mainhead.h"
/*
	Returns if the file is open (true) or closed (false) after call to function.
*/
bool printXYZV(DevDataBlock * dev_dat, HostDataBlock * host_dat, int Ntets, int Nnodes, float t, bool closeFile = false){
	static std::ofstream fout("Output//mv_director.xyzv", std::ios::out);
	if (!fout.is_open()) return false;
	if (closeFile){ fout.close(); return false; }

	//.. transfer data
	size_t height3 = 3;
	size_t widthNODE = Nnodes;

	HANDLE_ERROR(cudaMemcpy(host_dat->host_ThPhi, dev_dat->dev_ThPhi, Ntets*sizeof(int), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy2D(host_dat->host_r
		, widthNODE*sizeof(float)
		, dev_dat->dev_r
		, dev_dat->dev_rpitch
		, widthNODE*sizeof(float)
		, height3
		, cudaMemcpyDeviceToHost));

	fout << Ntets << std::endl;
	fout << "time=" << t << std::endl;
	float th, ph, x, y, z, nx, ny, nz;
	int n0, n1, n2, n3;
	for (int i = 0; i < Ntets; i++){

		//.. get theta and phi
		float thphi = float(host_dat->host_ThPhi[i]);
		float nTh = floor(thphi / 10000.0f);
		float nPh = thphi - nTh * 10000.0f;
		th = nTh*PI / 1000.0f;
		ph = nPh*PI / 500.0f;

		//.. director
		nx = cosf(ph)*sinf(th);
		ny = sinf(ph)*sinf(th);
		nz = cosf(th);

		//.. get nodes in tet
		n0 = host_dat->host_TetToNode[i + Ntets * 0];
		n1 = host_dat->host_TetToNode[i + Ntets * 1];
		n2 = host_dat->host_TetToNode[i + Ntets * 2];
		n3 = host_dat->host_TetToNode[i + Ntets * 3];

		//.. get position of tet
		x = (host_dat->host_r[n0 + 0 * Nnodes]
			+ host_dat->host_r[n1 + 0 * Nnodes]
			+ host_dat->host_r[n2 + 0 * Nnodes]
			+ host_dat->host_r[n3 + 0 * Nnodes]) / 4.0f;

		y = (host_dat->host_r[n0 + 1 * Nnodes]
			+ host_dat->host_r[n1 + 1 * Nnodes]
			+ host_dat->host_r[n2 + 1 * Nnodes]
			+ host_dat->host_r[n3 + 1 * Nnodes]) / 4.0f;

		z = (host_dat->host_r[n0 + 2 * Nnodes]
			+ host_dat->host_r[n1 + 2 * Nnodes]
			+ host_dat->host_r[n2 + 2 * Nnodes]
			+ host_dat->host_r[n3 + 2 * Nnodes]) / 4.0f;

		//.. print position and director
		fout << "A " << x/meshScale << " " << y/meshScale << " " << z/meshScale
			<< " " << nx << " " << ny << " " << nz << std::endl;
	}
	return true;
}