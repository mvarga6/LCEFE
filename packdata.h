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


	for (int tet = 0;tet<Ntets;tet++){
		dat->host_TetVol[tet] = i_Tet.get_volume(tet);
		dat->host_ThPhi[tet] = i_Tet.get_ThPhi(tet);
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
    if(i_Node.get_pos(nod,0)>39.0){
      dat->host_v[nod+Nnodes*1] = 10.0;
    }//if rx >39.0

		for(int rank=0;rank<MaxNodeRank;rank++){
		dat->host_dr[nod+rank]=0.0;
		}
	}//nod

	printf("Data packed to go to device\n");
}


#endif //__PACKDATA_H__
