#ifndef __PACKDATA_H__
#define __PACKDATA_H__
#include "parameters.h"
#include <math.h>
#include <vector>

//this function takes all the data about the simulatin and 
//packs it in a way that will make it easy to copy to GPU
void packdata(NodeArray &i_Node,TetArray &i_Tet, HostDataBlock *dat, int Ntets, int Nnodes,
		std::vector<int>* surf_Tets){

	// set the number of tets and nodes
	dat->Ntets = Ntets;
	dat->Nnodes = Nnodes;

	//allocate memory on host
	dat->A = (float*)malloc(Ntets*16*(sizeof(float)));
	dat->TetToNode = (int*)malloc(Ntets*4*(sizeof(int)));
	dat->r0 = (float*)malloc(Nnodes*3*(sizeof(float)));
	dat->r = (float*)malloc(Nnodes*3*(sizeof(float)));
	dat->F = (float*)malloc(Nnodes*3*(sizeof(float)));
	dat->v = (float*)malloc(Nnodes*3*(sizeof(float)));
	dat->nodeRank = (int*)malloc(Nnodes*sizeof(int));
	dat->m = (float*)malloc(Nnodes*sizeof(float));
	dat->pe = (float*)malloc(Ntets*sizeof(float));
	dat->TetNodeRank = (int*)malloc(Ntets*4*sizeof(int));
	dat->dr = (float*)malloc(Nnodes*MaxNodeRank*sizeof(float));
	dat->totalVolume = i_Tet.get_total_volume();
	dat->TetVol = (float*)malloc(Ntets*sizeof(float));
	dat->ThPhi = (int*)malloc(Ntets*sizeof(int));
	dat->S = (int*)malloc(Ntets*sizeof(int));

	//.. untransformed max's and min's
	float L;//, w, h;
	for(int c = 0; c < 3; c++){
		dat->min[c] = i_Node.min_point(c);
		dat->max[c] = i_Node.max_point(c);
	}
	L = dat->max[0] - dat->min[0];
//	w = dat->max[1] - dat->min[1];
//	h = dat->max[2] - dat->min[2];

	//.. determine tets on the top surface of film and build list
	float rz;
	for(int t = 0; t < Ntets; t++){ // for all tets
		rz = 0;
		for(int i = 0; i < 4; i++){ // tet neighbors (to get average z pos)
			rz += 0.25f * i_Node.get_pos(i_Tet.get_nab(t,i), 2); // z pos of node in tet
		}

		//.. condition to consider on surface (within one mesh unit of top surface)
		if(rz > (dat->max[2] - meshScale)) surf_Tets->push_back(t);
	}


	for (int tet = 0;tet<Ntets;tet++){
		dat->TetVol[tet] = i_Tet.get_volume(tet);
		dat->ThPhi[tet] = i_Tet.get_ThPhi(tet);
		dat->S[tet] = i_Tet.get_iS(tet);
		for (int sweep = 0;sweep<4;sweep++){

				dat->TetToNode[tet+sweep*Ntets] = i_Tet.get_nab(tet,sweep);
				dat->TetNodeRank[tet+sweep*Ntets] = i_Tet.get_nabRank(tet,sweep);

			//pack A inverted matrix
				for(int sweep2 = 0;sweep2<4;sweep2++){
			     dat->A[tet+(4*sweep+sweep2)*Ntets] = i_Tet.get_invA(tet,sweep,sweep2);
				}
		}//sweep
	}//tet

	for(int nod = 0;nod<Nnodes;nod++){
		dat->nodeRank[nod] = i_Node.get_totalRank(nod);
		dat->m[nod]=abs(i_Node.get_volume(nod)*materialDensity) ;

		for(int sweep = 0;sweep<3;sweep++){
			dat->r[nod+Nnodes*sweep] = i_Node.get_pos(nod,sweep);
			dat->r0[nod+Nnodes*sweep] = i_Node.get_pos(nod,sweep);
			dat->v[nod+Nnodes*sweep] = 0.0;
			dat->F[nod+Nnodes*sweep] = 0.0;
		}//sweep

    		//add force to end of beam
    		//if(i_Node.get_pos(nod,0)>39.9){
    		//  dat->host_v[nod+Nnodes*2] =100.0;
    		//}//if rx >39.0

		for(int rank=0;rank<MaxNodeRank;rank++){
			dat->dr[nod+rank]=0.0;
		}
	}//nod


	//.. transformation of initial state (leaves reference state intact)
	float x, z, minx=1000, maxx=0, minz=1000, maxz=0;
	for(int n = 0; n < Nnodes; n++){
		x = dat->r[n + Nnodes*0];
		z = dat->r[n + Nnodes*2];
		z += (SQZAMP * L) * sin(PI * (x - dat->min[0]) / L);
		x *= SQZRATIO;
		dat->r[n + Nnodes*0] = x;
		dat->r[n + Nnodes*2] = z;
		if(x > maxx) maxx = x;
		else if(x < minx) minx = x;
		if(z > maxz) maxz = z;
		else if(z < minz) minz = z;
	}
	dat->max[0] = maxx;
	dat->max[2] = maxz;
	dat->min[0] = minx;
	dat->min[2] = minz;

	printf("Data packed to go to device\n");
}


#endif //__PACKDATA_H__
