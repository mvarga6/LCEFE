#ifndef __ILLUMINATION_CPU_H__
#define __ILLUMINATION_CPU_H__

#include "mainhead.h"
#include <vector>
#include <algorithm>
#include <map>
#include <string>

struct adrs_tet_pos {
	int adrs;
	int tet;
	float posx, posy, posz;
	bool operator < (const adrs_tet_pos& cmp) const{
		return (adrs < cmp.adrs);
	}
};

bool sort_on_adrs(adrs_tet_pos a, adrs_tet_pos b) {
	return (a.adrs<b.adrs);
}

float dist_point_to_plane(float r[3], float n[3], float d){
	const float numer = n[0]*r[0]+n[1]*r[1]+n[2]*r[2]+d;
	const float denom = sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2]);
	return (numer/denom);
}

//=============================================================
//  On CPU (every print), calculate and set S for each tet

void calc_S_from_light(float k[3], float *r, int *TetToNode, int Ntets, int Nnodes, int *S, float cell_dx, float cell_dy){

	//.. allocate whats needed
	int * illum_cell = new int[Ntets*2]; // { i1, j1, i2, j2, ... , iN, jN }
	std::vector<adrs_tet_pos> tetData(Ntets); // maps node to cell address
	int max_i = -100000, min_i = 100000, max_j = -100000, min_j = 100000; // range of cells
	int icell, jcell;
	printf("\nScanning through all %d  tetrahedras...\n",Ntets);

	//.. first, calculate where light eminates from on plane wave (which cell with coords)
	for(int t = 0; t < Ntets; t++){ // for all tets
		float _r[12]; 
		float _rcom[3] = { 0, 0, 0 }; // position of nodes and tet c.o.m
		int mynode;
		for(int n = 0; n < 4; n++){ // for nodes 1,2,3,4
			mynode = TetToNode[t + n*Ntets]; // get node n of tet t
			for(int c = 0; c < 3; c++){ // for x,y,z
				_r[c+n*3] = r[mynode + c*Nnodes]; 
				_rcom[c] += _r[c+n*3] * 0.25f;  // implicit averaging
			}
		}

		//.. store tet ave pos 
		tetData.at(t).posx = _rcom[0];
		tetData.at(t).posy = _rcom[1];
		tetData.at(t).posz = _rcom[2];

		//.. rotate com vector to place incident light along -z_hat
		const float k_mag = sqrt(k[0]*k[0]+k[1]*k[1]+k[2]*k[2]);
		const float k_mag_xy = sqrt(k[0]*k[0]+k[1]*k[1]);
		const float phi = atan2(-k[1], -k[0]);  // angle in xy plane
		const float the = PI - atan2(-k[2], k_mag_xy); // angle from z axis
		const float cphi = cos(phi);
		const float sphi = sin(phi);
		const float cthe = cos(the);
		const float sthe = sin(the);
		float R[3][3];
		R[0][0] = cthe*cphi;
		R[0][1] = -cthe*sphi;
		R[0][2] = -sthe;
		R[1][0] = sphi;
		R[1][1] = cphi;
		R[1][2] = 0;
		R[2][0] = sthe*cphi;
		R[2][1] = -sthe*sphi;
		R[2][2] = cthe;

		float _rcomp[3] = {0, 0, 0};
		float jsum;
		for(int i = 0; i < 3; i++){ // matrix multiplication r'[i] = R[i][j]*r[j]
			jsum = 0;
			for(int j = 0; j < 3; j++){
				jsum += R[i][j]*_rcom[j];
			}
			_rcomp[i] = jsum;
		}

		icell = int(floor(_rcomp[0] / cell_dx));
		jcell = int(floor(_rcomp[2] / cell_dy));
		illum_cell[t + 0] = icell;
		illum_cell[t + 1] = jcell;
		if(icell > max_i) max_i = icell; // store max/min
		if(icell < min_i) min_i = icell;
		if(jcell > max_j) max_j = jcell;
		if(jcell < min_j) min_j = jcell;
		printf("\n%d\t%d\t%d",t,icell,jcell);
	}

	printf("\n--------------------");
	printf("\nmax\t%d\t%d",max_i,max_j);
	printf("\nmin\t%d\t%d",min_i,min_j);
	printf("\nrange\t%d\t%d",max_i-min_i,max_j-min_j);

	printf("\nDONE");
	//printf("\ni: %d -> %d\nj: %d -> %d", min_i, max_i, min_j, max_j); 
	printf("\nLabelling all addresses...");
	//.. shift to only positive indices
	const int width = (max_i - min_i); 
	const int height = (max_j - min_j);
	printf("\nIncident light grid dimensions: %d x %d", width, height);
	for(int t = 0; t < Ntets; t++){
		illum_cell[t + 0] -= min_i; // shift to start at zero
		illum_cell[t + 1] -= min_j; // shift to start at zero
		tetData.at(t).adrs = illum_cell[t + 0] + width*illum_cell[t + 1]; // make address
		tetData.at(t).tet = t;
	}
	delete [] illum_cell;

	printf("\tDONE");

	//.. sort address to tet map on address, this puts all tets
	//   with same illumination origin next to eachother
	std::sort(tetData.begin(), tetData.end());

	//.. mark only the closest tets (per unique illumination origin) as lit
	//std::vector<adrs_tet_pos>::iterator _end = tetData.end(); 
	//std::vector<adrs_tet_pos>::iterator it = tetData.begin();
	//while(it != _end){
	for (int i = 0; i < Ntets - 1;){
		
		//.. setup queue pointing to all tetData with same adrs
		std::vector<adrs_tet_pos> queue; // ptrs to data
		
		do {
			queue.push_back(tetData[i++]);
			if (i >= Ntets) break;
		} while (tetData[i-1].adrs == tetData[i].adrs);
			
		//do { 
		// queue.push_back(&(*it)); // add reference to where it points to queue
		//} while(((*it).adrs == (*(it+1)).adrs) && (((it++)+1) != _end)); // continue if next element has same address and exists

		//.. find closed tet in queue
		int closest_tet; float min_dist = 1000000;
		float dist;
		for(int i = 0; i < queue.size(); i++){
			float pos[3] = { queue.at(i)->posx, queue.at(i)->posy, queue.at(i)->posz};
		int closest_tet; 
		float min_dist = 1000000;
		float dist;
		for(int q = 0; q < queue.size(); q++){
			float pos[3] = { queue[q].posx, queue[q].posy, queue[q].posz};
			dist = dist_point_to_plane(pos , k, 1000);
			if(dist < min_dist) closest_tet = queue[q].tet;
		}

		S[closest_tet] = 0;
	}

	// DONE !
	printf("\nS adjusted based on illumination");
}

#endif
