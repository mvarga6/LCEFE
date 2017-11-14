#include "../datastruct.h"


HostDataBlock::HostDataBlock(NodeArray* Nodes, TetArray *Tets, SimulationParameters *params)
{
	int Ntets = Tets->size;
	int Nnodes = Nodes->size;

	// set the number of tets and nodes
	this->Ntets = Ntets;
	this->Nnodes = Nnodes;
	
	//allocate memory on host
	this->A 			 = (real*)malloc(Ntets*16*(sizeof(real)));
	this->TetToNode 	 = (int*)malloc(Ntets*4*(sizeof(int)));
	this->r0 		 = (real*)malloc(Nnodes*3*(sizeof(real)));
	this->r 			 = (real*)malloc(Nnodes*3*(sizeof(real)));
	this->F 			 = (real*)malloc(Nnodes*3*(sizeof(real)));
	this->v 			 = (real*)malloc(Nnodes*3*(sizeof(real)));
	this->nodeRank 	 = (int*)malloc(Nnodes*sizeof(int));
	this->m 		 	 = (real*)malloc(Nnodes*sizeof(real));
	this->pe 		 = (real*)malloc(Ntets*sizeof(real));
	this->TetNodeRank = (int*)malloc(Ntets*4*sizeof(int));
	this->dr 		 = (real*)malloc(Nnodes*MaxNodeRank*sizeof(real));
	this->totalVolume = Tets->get_total_volume();
	this->TetVol 	 = (real*)malloc(Ntets*sizeof(real));
	this->ThPhi 		 = (int*)malloc(Ntets*sizeof(int));
	this->S 			 = (int*)malloc(Ntets*sizeof(int));
	
	//.. untransformed max's and min's
	//real L;//, w, h;
	for(int c = 0; c < 3; c++)
	{
		this->min[c] = Nodes->min_point(c);
		this->max[c] = Nodes->max_point(c);
	}

	//.. determine tets on the top surface of film and build list
	real rz;
	for(int t = 0; t < Ntets; t++)
	{ // for all tets
		rz = 0;
		for(int i = 0; i < 4; i++)
		{ // tet neighbors (to get average z pos)
			rz += 0.25f * Nodes->get_pos(Tets->get_nab(t,i), 2); // z pos of node in tet
		}
	}


	for (int tet = 0; tet < Ntets; tet++)
	{
		this->TetVol[tet] = Tets->get_volume(tet);
		this->ThPhi[tet] = Tets->get_ThPhi(tet);
		this->S[tet] = Tets->get_iS(tet);
		for (int sweep = 0; sweep < 4; sweep++)
		{
			this->TetToNode[tet+sweep*Ntets] = Tets->get_nab(tet,sweep);
			this->TetNodeRank[tet+sweep*Ntets] = Tets->get_nabRank(tet,sweep);

			//pack A inverted matrix
			for(int sweep2 = 0; sweep2 < 4; sweep2++)
			{
				this->A[tet+(4*sweep+sweep2)*Ntets] = Tets->get_invA(tet,sweep,sweep2);
			}
		}//sweep
	}//tet

	for(int nod = 0;nod < Nnodes; nod++){
		this->nodeRank[nod] = Nodes->get_totalRank(nod);
		this->m[nod]=abs(Nodes->get_volume(nod) * params->Material.Density) ;

		for(int sweep = 0; sweep < 3; sweep++)
		{
			this->r[nod+Nnodes*sweep] = Nodes->get_pos(nod,sweep);
			this->r0[nod+Nnodes*sweep] = Nodes->get_pos(nod,sweep);
			this->v[nod+Nnodes*sweep] = 0.0;
			this->v[nod+Nnodes*sweep] = 0.0; //10.f*(genrand() - 0.5f);
			this->F[nod+Nnodes*sweep] = 0.0; //100.f*(genrand() - 0.5f);
		}//sweep

		for(int rank = 0; rank < MaxNodeRank; rank++)
		{
			this->dr[nod+rank] = 0.0;
		}
	}//nod


	//.. transformation of initial state (leaves reference state intact)
	// TODO: Initiall state transformations before packthisa
}
