#ifndef __PACKDATA_H__
#define __PACKDATA_H__

#include <math.h>
#include <vector>
#include "parameters.h"
#include "simulation_parameters.h"
#include "genrand.h"

//this function takes all the data about the simulatin and 
//packs it in a way that will make it easy to copy to GPU
void packdata(NodeArray &Nodes,TetArray &Tets, HostDataBlock *dat, 
	std::vector<int>* surf_Tets, SimulationParameters *params)
{
	int Ntets = Tets.size;
	int Nnodes = Nodes.size;

	// set the number of tets and nodes
	dat->Ntets = Ntets;
	dat->Nnodes = Nnodes;

	//allocate memory on host
	dat->A 			 = (real*)malloc(Ntets*16*(sizeof(real)));
	dat->TetToNode 	 = (int*)malloc(Ntets*4*(sizeof(int)));
	dat->r0 		 = (real*)malloc(Nnodes*3*(sizeof(real)));
	dat->r 			 = (real*)malloc(Nnodes*3*(sizeof(real)));
	dat->F 			 = (real*)malloc(Nnodes*3*(sizeof(real)));
	dat->v 			 = (real*)malloc(Nnodes*3*(sizeof(real)));
	dat->nodeRank 	 = (int*)malloc(Nnodes*sizeof(int));
	dat->m 		 	 = (real*)malloc(Nnodes*sizeof(real));
	dat->pe 		 = (real*)malloc(Ntets*sizeof(real));
	dat->TetNodeRank = (int*)malloc(Ntets*4*sizeof(int));
	dat->dr 		 = (real*)malloc(Nnodes*MaxNodeRank*sizeof(real));
	dat->totalVolume = Tets.get_total_volume();
	dat->TetVol 	 = (real*)malloc(Ntets*sizeof(real));
	dat->ThPhi 		 = (int*)malloc(Ntets*sizeof(int));
	dat->S 			 = (int*)malloc(Ntets*sizeof(int));

	//.. untransformed max's and min's
	//real L;//, w, h;
	for(int c = 0; c < 3; c++)
	{
		dat->min[c] = Nodes.min_point(c);
		dat->max[c] = Nodes.max_point(c);
	}

	//.. determine tets on the top surface of film and build list
	real rz;
	for(int t = 0; t < Ntets; t++)
	{ // for all tets
		rz = 0;
		for(int i = 0; i < 4; i++)
		{ // tet neighbors (to get average z pos)
			rz += 0.25f * Nodes.get_pos(Tets.get_nab(t,i), 2); // z pos of node in tet
		}
	}


	for (int tet = 0; tet < Ntets; tet++)
	{
		dat->TetVol[tet] = Tets.get_volume(tet);
		dat->ThPhi[tet] = Tets.get_ThPhi(tet);
		dat->S[tet] = Tets.get_iS(tet);
		for (int sweep = 0; sweep < 4; sweep++)
		{
			dat->TetToNode[tet+sweep*Ntets] = Tets.get_nab(tet,sweep);
			dat->TetNodeRank[tet+sweep*Ntets] = Tets.get_nabRank(tet,sweep);

			//pack A inverted matrix
			for(int sweep2 = 0; sweep2 < 4; sweep2++)
			{
				dat->A[tet+(4*sweep+sweep2)*Ntets] = Tets.get_invA(tet,sweep,sweep2);
			}
		}//sweep
	}//tet

	for(int nod = 0;nod < Nnodes; nod++){
		dat->nodeRank[nod] = Nodes.get_totalRank(nod);
		dat->m[nod]=abs(Nodes.get_volume(nod) * params->Material.Density) ;

		for(int sweep = 0; sweep < 3; sweep++)
		{
			dat->r[nod+Nnodes*sweep] = Nodes.get_pos(nod,sweep);
			dat->r0[nod+Nnodes*sweep] = Nodes.get_pos(nod,sweep);
			dat->v[nod+Nnodes*sweep] = 0.0;
			dat->v[nod+Nnodes*sweep] = 0.0; //10.f*(genrand() - 0.5f);
			dat->F[nod+Nnodes*sweep] = 0.0; //100.f*(genrand() - 0.5f);
		}//sweep

		for(int rank = 0; rank < MaxNodeRank; rank++)
		{
			dat->dr[nod+rank] = 0.0;
		}
	}//nod


	//.. transformation of initial state (leaves reference state intact)
	// TODO: Initiall state transformations before packdata
}


#endif //__PACKDATA_H__
