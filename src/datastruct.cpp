#include "datastruct.h"
#include "errorhandle.h"
#include "defines.h"

HostDataBlock::HostDataBlock(NodeArray* Nodes, TetArray *Tets, TriArray *Tris, SimulationParameters *params)
{
	int Ntets = Tets->size;
	int Nnodes = Nodes->size;
	int Ntris = Tris->size;

	// set the number of tets and nodes
	this->Ntets = Ntets;
	this->Nnodes = Nnodes;
	this->Ntris = Ntris;
	
	//allocate memory on host
	this->A 			 = (real*)malloc(Ntets*16*(sizeof(real)));
	this->TetToNode 	 = (int*)malloc(Ntets*4*(sizeof(int)));
	this->TriToNode		 = (int*)malloc(Ntris*3*sizeof(int));
	this->r0 		 = (real*)malloc(Nnodes*3*(sizeof(real)));
	this->r 			 = (real*)malloc(Nnodes*3*(sizeof(real)));
	this->F 			 = (real*)malloc(Nnodes*3*(sizeof(real)));
	this->v 			 = (real*)malloc(Nnodes*3*(sizeof(real)));
	this->nodeRank 	 = (int*)malloc(Nnodes*sizeof(int));
	this->nodeRankWrtTris = (int*)malloc(Nnodes*sizeof(int));
	this->m 		 	 = (real*)malloc(Nnodes*sizeof(real));
	this->pe 		 = (real*)malloc(Ntets*sizeof(real));
	this->TetNodeRank = (int*)malloc(Ntets*4*sizeof(int));
	this->TriNodeRank = (int*)malloc(Ntris*3*sizeof(int));
	this->dr 		 = (real*)malloc(Nnodes*MaxNodeRank*sizeof(real));
	this->totalVolume = Tets->get_total_volume();
	this->TargetEnclosedVolume = 0;
	this->TetVol 	 = (real*)malloc(Ntets*sizeof(real));
	this->TriArea	 = (real*)malloc(Ntris*sizeof(real));
	this->TriNormal 	= (real*)malloc(Ntris*3*sizeof(real));
	this->ThPhi 		 = (int*)malloc(Ntets*sizeof(int));
	this->S 			 = (real*)malloc(Ntets*sizeof(real));
	
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

	///
	/// Pack data from tets
	///
	
	for (int tet = 0; tet < Ntets; tet++)
	{
		this->TetVol[tet] = Tets->get_volume(tet);
		this->ThPhi[tet] = Tets->get_ThPhi(tet);
		this->S[tet] = Tets->get_S(tet);
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


	///
	/// Pack data for Tris
	///
	for (int tri = 0; tri < Ntris; tri++)
	{
		this->TriArea[tri] = Tris->area(tri);
		
		for (int j = 0; j < 3; j++)
		{
			this->TriToNode[tri + j*Ntris] = Tris->node_idx(tri, j);
			this->TriNodeRank[tri + j*Ntris] = Tris->rank(tri, j);
			this->TriNormal[tri + j*Ntris] = Tris->normal(tri, j);
		}
	}

	///
	/// Pack data for Nodes
	///
	for(int nod = 0;nod < Nnodes; nod++){
		this->nodeRank[nod] 	    = Nodes->get_rank_wrt_tets(nod);
		this->nodeRankWrtTris[nod] 	= Nodes->get_rank_wrt_tris(nod);
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

DevDataBlock* HostDataBlock::CreateDevDataBlock()
{
	// new up data block
	DevDataBlock *dev = new DevDataBlock();

	//need to pitch 1D memory correctly to send to device
	int Nnodes = this->Nnodes;
	int Ntets = this->Ntets;
	int Ntris = this->Ntris;
	size_t height16 = 16;
	size_t height4 = 4;
	size_t height3 = 3;
	size_t heightMR = MaxNodeRank*3;
	size_t widthNODE = Nnodes;
	size_t widthTETS = Ntets;
	
	dev->Nnodes = Nnodes;
	dev->Ntets = Ntets;
	dev->Ntris = Ntris;

	//set offset to be 0
	//size_t offset = 0;

	//used pitch linear memory on device for fast access
	//allocate memory on device for pitched linear memory

	///
	/// NODES
	///

	HANDLE_ERROR( cudaMallocPitch( (void**) &dev->r0 
		, &dev->r0pitch  
		, widthNODE*sizeof(real) 
		, height3 ) );
	HANDLE_ERROR( cudaMallocPitch( (void**) &dev->r 
		, &dev->rpitch 
		, widthNODE*sizeof(real) 
		, height3 ) );
	HANDLE_ERROR( cudaMallocPitch( (void**) &dev->v 
		, &dev->vpitch 
		, widthNODE*sizeof(real) 
		, height3 ) );
	HANDLE_ERROR( cudaMallocPitch( (void**) &dev->F
		, &dev->Fpitch
		, widthNODE*sizeof(real) 
		, height3 ) );
	HANDLE_ERROR( cudaMallocPitch( (void**) &dev->dF 
		, &dev->dFpitch 
		, widthNODE*sizeof(real) 
		, heightMR ) );

	printf("\n[ INFO ] r0pitch = %d", dev->r0pitch);
	printf("\n[ INFO ] rpitch = %d", dev->rpitch);
	printf("\n[ INFO ] vpitch = %d", dev->vpitch);
	printf("\n[ INFO ] Fpitch = %d", dev->Fpitch);
	printf("\n[ INFO ] dFpitch = %d", dev->dFpitch);

	
	HANDLE_ERROR( cudaMalloc( (void**) &dev->nodeRankWrtTris, Nnodes*sizeof(int) ) );
	HANDLE_ERROR( cudaMalloc( (void**) &dev->nodeRank, Nnodes*sizeof(int) ) );
	HANDLE_ERROR( cudaMalloc( (void**) &dev->m, Nnodes*sizeof(real) ) );

	///
	/// TETRAHEDRA
	///

	HANDLE_ERROR( cudaMallocPitch( (void**) &dev->A 
									, &dev->Apitch 
									, widthTETS*sizeof(real) 
									, height16 ) );

	

	// HANDLE_ERROR( cudaMallocPitch( (void**) &dev->F_tri 
	// 								, &dev->Ftripitch 
	// 								, widthNODE*sizeof(real) 
	// 								, heightMR ) );

	HANDLE_ERROR( cudaMallocPitch( (void**) &dev->TetToNode 
									, &dev->TetToNodepitch 
									, widthTETS*sizeof(int) 
									, height4 ) );

	HANDLE_ERROR( cudaMallocPitch((void**) &dev->TetNodeRank
									, &dev->TetNodeRankpitch
									, widthTETS*sizeof(int)
									, height4 ) );

	printf("\n[ INFO ] Apitch = %d", dev->Apitch);
	printf("\n[ INFO ] TetToNodepitch = %d", dev->TetToNodepitch);
	printf("\n[ INFO ] TetNodeRankpitch = %d", dev->TetNodeRankpitch);
	
	//HANDLE_ERROR( cudaMalloc( (void**) &dev->TetNodeRank, Ntets*4*sizeof(int) ) );
	HANDLE_ERROR( cudaMalloc( (void**) &dev->pe, Ntets*sizeof(real) ) );
	HANDLE_ERROR( cudaMalloc( (void**) &dev->TetVol, Ntets*sizeof(real) ) );
	HANDLE_ERROR( cudaMalloc( (void**) &dev->ThPhi, Ntets*sizeof(int) ) );
	HANDLE_ERROR( cudaMalloc( (void**) &dev->S, Ntets*sizeof(real) ) );
	HANDLE_ERROR( cudaMalloc( (void**) &dev->L, Ntets*sizeof(int) ) );
	

	///
	/// TRIANGLES
	///

	HANDLE_ERROR( cudaMallocPitch((void**) &dev->TriToNode
									, &dev->TriToNodepitch
									, Ntris * sizeof(int)
									, 3) );

	HANDLE_ERROR( cudaMallocPitch((void**) &dev->TriNodeRank
									, &dev->TriNodeRankpitch
									, Ntris*sizeof(int)
									, 3 ) );

	HANDLE_ERROR( cudaMallocPitch((void**) &dev->TriNormal
									, &dev->TriNormalpitch
									, Ntris*sizeof(real)
									, 3 ) );

	printf("\n[ INFO ] TriToNodepitch = %d", dev->TriToNodepitch);
	printf("\n[ INFO ] TriNodeRankpitch = %d", dev->TriNodeRankpitch);
	printf("\n[ INFO ] TetNormalpitch = %d", dev->TriNormalpitch);

	HANDLE_ERROR( cudaMalloc( (void**) &dev->EnclosedVolume, Ntris * sizeof(real) ) );
	HANDLE_ERROR( cudaMalloc( (void**) &dev->TriArea, Ntris * sizeof(real) ) );

	return dev;
}

PointerHandle<real> DevDataBlock::HandleForS()
{
	return GpuPointerHandle<real>(this->S, this->Ntets);
}

PointerHandle<int> DevDataBlock::HandleForDirector()
{
	return GpuPointerHandle<int>(this->ThPhi, this->Ntets);
}