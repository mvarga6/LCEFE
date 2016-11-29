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

struct adrs_node_tets {
	int adrs, node;
	std::vector<int> tets;
	float x, y, z;
	bool operator < (const adrs_node_tets& cmp) const{
		return (adrs < cmp.adrs);
	}
};

bool sort_on_adrs(adrs_tet_pos a, adrs_tet_pos b) {
	return (a.adrs<b.adrs);
}

float dist_point_to_plane(float _r[3], float _n[3], float d){
	const float numer = abs(_n[0]*_r[0]+_n[1]*_r[1]+_n[2]*_r[2]+d);
	const float denom = sqrt(_n[0]*_n[0]+_n[1]*_n[1]+_n[2]*_n[2]);
	return numer/denom;
}

//=============================================================
//  On CPU (every print), calculate and set S for each tet

void calc_S_from_light(float k[3], float *r, int *TetToNode, 
			int Ntets, int Nnodes, int *S,
			std::vector<int>* tetsList, 
			float cell_dx, float cell_dy){

	//.. allocate whats needed
	int Nsurf = tetsList->size();
	int * illum_cell = new int[Nsurf*2]; // { i1, j1, i2, j2, ... , iN, jN }
	std::vector<adrs_tet_pos> data(Nsurf); // maps node to cell address
	int max_i = -100000, min_i = 100000, max_j = -100000, min_j = 100000; // range of cells
	int icell, jcell;

	//.. calc rotation matrix to place incident light along -z_hat
	const float k_mag = sqrt(k[0]*k[0]+k[1]*k[1]+k[2]*k[2]);
	const float k_mag_xy = sqrt(k[0]*k[0]+k[1]*k[1]);
	const float phi = atan2(k[1], k[0]);  // angle in xy plane
	const float the = atan2(k_mag_xy, -k[2]); // angle from z axis
	const float cphi = cos(phi);
	const float sphi = sin(phi);
	const float cthe = cos(the);
	const float sthe = sin(the);
	float R[3][3];
	R[0][0] = cthe*cphi;
	R[0][1] = -cthe*sphi;
	R[0][2] = sthe;
	R[1][0] = sphi;
	R[1][1] = cphi;
	R[1][2] = 0;
	R[2][0] = -sthe*cphi;
	R[2][1] = sthe*sphi;
	R[2][2] = cthe;

	//.. check what new k vector is
	float kp[3];
	kp[0] = k[0]*R[0][0] + k[1]*R[0][1] + k[2]*R[0][2];
	kp[1] = k[0]*R[1][0] + k[1]*R[1][1] + k[2]*R[1][2];
	kp[2] = k[0]*R[2][0] + k[1]*R[2][1] + k[2]*R[2][2];
	printf("\nk  = { %f, %f, %f }", k[0], k[1], k[2]);
	printf("\nkp = { %f, %f, %f }", kp[0], kp[1], kp[2]);

	//.. first, calculate where light eminates from on plane wave (which cell with coords)
	for(int _t = 0; _t < Nsurf; _t++){ // for all tets on in list
		int t = tetsList->at(_t);
		float _r[12]; 
		float _rcom[3] = { 0, 0, 0 }; // position of nodes and tet c.o.m
		int mynode;
		for(int n = 0; n < 4; n++){ // for nodes 1,2,3,4
			mynode = TetToNode[t + n*Ntets]; // get node n of tet t
			for(int c = 0; c < 3; c++){ // for x,y,z
				_r[n+c*3] = r[mynode + c*Nnodes]; 
				_rcom[c] += _r[n+c*3] * 0.25f;  // implicit averaging
			}
		}

		//.. store tet ave pos 
		data.at(_t).posx = _rcom[0];
		data.at(_t).posy = _rcom[1];
		data.at(_t).posz = _rcom[2];

		//.. Rotate incident light to -k_hat
		float _rcomp[3] = {0, 0, 0}, jsum;
		for(int i = 0; i < 3; i++){ // matrix multiplication r'[i] = R[i][j]*r[j]
			jsum = 0;
			for(int j = 0; j < 3; j++){
				jsum += R[i][j]*_rcom[j];
			}
			_rcomp[i] = jsum;
		}

		icell = int(floor(_rcomp[0] / cell_dx));
		jcell = int(floor(_rcomp[1] / cell_dy));
		illum_cell[_t + 0] = icell;
		illum_cell[_t + 1] = jcell;
		if(icell > max_i) max_i = icell; // store max/min
		if(icell < min_i) min_i = icell;
		if(jcell > max_j) max_j = jcell;
		if(jcell < min_j) min_j = jcell;
	}

	printf("\n\nmax\t%d\t%d",max_i,max_j);
	printf("\nmin\t%d\t%d",min_i,min_j);
	printf("\nrange\t%d\t%d",max_i-min_i,max_j-min_j);

	//.. shift to only positive indices
	const int width = (max_i - min_i); 
	const int height = (max_j - min_j);
	printf("\nIncident light grid dimensions: %d x %d", width, height);
	for(int _t = 0; _t < Nsurf; _t++){
		illum_cell[_t + 0] -= min_i; // shift to start at zero
		illum_cell[_t + 1] -= min_j; // shift to start at zero
		data.at(_t).adrs = illum_cell[_t + 0] + width*illum_cell[_t + 1]; // make address
		data.at(_t).tet = tetsList->at(_t); // need to store actual tet idx
	}
	delete [] illum_cell;

	printf("\tDONE");

	//.. sort address to tet map on address, this puts all tets
	//   with same illumination origin next to eachother
	std::sort(data.begin(), data.end());

	//.. mark only the closest tets (per unique illumination origin) as lit
	int count = 0;
	for (int _t = 0; _t < Nsurf - 1; /*inc in queue set*/){

		//.. setup queue pointing to all tetData with same adrs
		std::vector<adrs_tet_pos> queue; // ptrs to data
		int que_size = 0;
		do {
			queue.push_back(data[_t++]);
			que_size++;
			if (_t >= Nsurf) break;
		} while (data.at(_t-1).adrs == data.at(_t).adrs);

		//.. find closed node in queue
		int closest_tet; float min_dist = 1000000;
		float dist;
		for(int q = 0; q < que_size; q++){
			float pos[3] = { queue[q].posx, queue[q].posy, queue[q].posz };
			dist = dist_point_to_plane(pos, k, 100);
			if(dist < min_dist){
				closest_tet = queue[q].tet; 
				min_dist = dist;
			}
		}
		count++;

		//.. adjust order parameter for closest
		int tet;
		for(int q = 0; q < que_size; q++){
			tet = queue[q].tet;
			if(tet == closest_tet) // closed tet lowers
				S[tet] += int(SRES*SRATE_ON);
			else
				S[tet] += int(SRES*SRATE_OFF); // all other raise

			if(S[tet] > SMAX*SRES) 
				S[tet] = int(SMAX*SRES); // constrain to S range
			else if (S[tet] < SMIN*SRES) 
				S[tet] = int(SMIN*SRES);
		}
	}

	// DONE !
	printf("\n%d%c of top surface tets illuminated",int(100*float(count)/float(Ntets)),'%');
}

#endif
