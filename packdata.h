#ifndef __PACKDATA_H__
#define __PACKDATA_H__
#include "parameters.h"
#include <math.h>


//this function takes all the data about the simulatin and 
//packs it in a way that will make it easy to copy to GPU
void packdata(NodeArray &i_Node,TetArray &i_Tet, HostDataBlock *dat,int Ntets,int Nnodes){

	//allocate memory on host
	dat->host_A = (float*)malloc(Ntets*16*(sizeof(float)));
	dat->host_TetToNode = (int*)malloc(Ntets*4*(sizeof(int)));
	dat->host_r0 = (float*)malloc(Nnodes*3*(sizeof(float)));
	dat->host_r = (float*)malloc(Nnodes*3*(sizeof(float)));
	dat->host_F = (float*)malloc(Nnodes*3*(sizeof(float)));
	dat->host_v = (float*)malloc(Nnodes*3*(sizeof(float)));
	dat->host_nodeRank = (int*)malloc(Nnodes*sizeof(int));
	dat->host_m = (float*)malloc(Nnodes*sizeof(float));
	dat->host_pe = (float*)malloc(Ntets*sizeof(float));
	dat->host_TetNodeRank = (int*)malloc(Ntets*4*sizeof(int));
	dat->host_dr = (float*)malloc(Nnodes*MaxNodeRank*sizeof(float));
	dat->host_totalVolume = i_Tet.get_total_volume();
	dat->host_TetVol = (float*)malloc(Ntets*sizeof(float));
	dat->host_ThPhi = (int*)malloc(Ntets*sizeof(int));
	dat->host_S = (int*)malloc(Ntets*sizeof(int));

	float L;//, w, h;
	for(int c = 0; c < 3; c++){
		dat->min[c] = i_Node.min_point(c);
		dat->max[c] = i_Node.max_point(c);
	}
	L = dat->max[0] - dat->min[0];
//	w = dat->max[1] - dat->min[1];
//	h = dat->max[2] - dat->min[2];

	for (int tet = 0;tet<Ntets;tet++){
		dat->host_TetVol[tet] = i_Tet.get_volume(tet);
		dat->host_ThPhi[tet] = i_Tet.get_ThPhi(tet);
		dat->host_S[tet] = i_Tet.get_iS(tet);
		for (int sweep = 0;sweep<4;sweep++){

				dat->host_TetToNode[tet+sweep*Ntets] = i_Tet.get_nab(tet,sweep);
				dat->host_TetNodeRank[tet+sweep*Ntets] = i_Tet.get_nabRank(tet,sweep);

			//pack A inverted matrix
				for(int sweep2 = 0;sweep2<4;sweep2++){
			     dat->host_A[tet+(4*sweep+sweep2)*Ntets] = i_Tet.get_invA(tet,sweep,sweep2);
				}
		}//sweep
	}//tet

	for(int nod = 0;nod<Nnodes;nod++){
		dat->host_nodeRank[nod] = i_Node.get_totalRank(nod);
		dat->host_m[nod]=abs(i_Node.get_volume(nod)*materialDensity) ;

		for(int sweep = 0;sweep<3;sweep++){
			dat->host_r[nod+Nnodes*sweep] = i_Node.get_pos(nod,sweep);
			dat->host_r0[nod+Nnodes*sweep] = i_Node.get_pos(nod,sweep);
			dat->host_v[nod+Nnodes*sweep] = 0.0;
			dat->host_F[nod+Nnodes*sweep] = 0.0;
		}//sweep

    		//add force to end of beam
    		//if(i_Node.get_pos(nod,0)>39.9){
    		//  dat->host_v[nod+Nnodes*2] =100.0;
    		//}//if rx >39.0

		for(int rank=0;rank<MaxNodeRank;rank++){
			dat->host_dr[nod+rank]=0.0;
		}
	}//nod

	

	//.. transformation of initial state (leaves reference state intact)
	float x, z, minx=1000, maxx=0, minz=1000, maxz=0;
	for(int n = 0; n < Nnodes; n++){
		x = dat->host_r[n + Nnodes*0];
		z = dat->host_r[n + Nnodes*2];
		z += (SQZAMP * L) * sin(PI * (x - dat->min[0]) / L);
		x *= SQZRATIO;
		dat->host_r[n + Nnodes*0] = x;
		dat->host_r[n + Nnodes*2] = z;
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
