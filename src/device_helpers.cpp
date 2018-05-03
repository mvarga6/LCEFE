#include "device_helpers.h"
#include "texdef.h"
#include "defines.h"

__device__
void DeviceHelpers::ReadGlobalToLocal(
	int (&node_num_local)[4]
	,int (&tet_node_rank_local)[4]
	,real (&Ainv_local)[16]
	,real (&r0_local)[12]
	,real (&r_local)[12]
	,real (&v_local)[12]
	,real *Ainv_global
	,int Ashift
	,real *r0_global
	,real *r_global
	,real *v_global
	,int vshift
	,int *tet_node_rank_global
	,int *tet_to_node_global
	,int TTNshift
	,int Ntets
	,int Nnodes
)
{
	int node_idx;
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	for(int n = 0; n < 4; n++)
	{
		//figure out what 4 nodes make up tetrahedra
		node_idx = tet_to_node_global[TTNshift*n + tid];
		tet_node_rank_local[n] = tet_node_rank_global[Ntets*n + tid];
		
		node_num_local[n] = node_idx;
		
		for(int cord = 0; cord < 4; cord++)
		{
			if(cord < 3)
			{
				//get orignal positions
				r0_local[cord+n*3] = r0_global[Nnodes*cord + node_idx];
				r_local[cord+n*3]  = r_global[Nnodes*cord + node_idx];
				v_local[cord+n*3]  = v_global[Nnodes*cord + node_idx];
			}//cord<3
					//get values of Ainv
			Ainv_local[cord+n*4] = Ainv_global[Ashift*(cord+n*4) + tid];
		}//cord
	}//n
}
	
__device__
void DeviceHelpers::SendForce(
	 real *dF
	,int dFshift
	,real F[12]
	,int NodeNum[4]
	,int TetNodeRank[4]
	,real TetVol,
	int tid
)
{
	int n_glob, NodeRank;  

	//loop over each node in tetrahedra
	for (int n = 0; n < 4; n++)
	{ 
		//find real node number
		n_glob = NodeNum[n];    
		NodeRank = TetNodeRank[n];

		dF[dFshift*(0+3*NodeRank) + n_glob] = F[0+3*n] * TetVol;
		dF[dFshift*(1+3*NodeRank) + n_glob] = F[1+3*n] * TetVol;
		dF[dFshift*(2+3*NodeRank) + n_glob] = F[2+3*n] * TetVol;
	}
	
#ifdef __DEBUG_SEND_FORCE__
		if (tid == __DEBUG_SEND_FORCE__)
		{
			printf("\n\n -- DeviceHelpers::SendForce --");
			printf("\n\tNode ids:\t[%d %d %d %d]", NodeNum[0], NodeNum[1], NodeNum[2], NodeNum[3]);
			printf("\n\tNode ranks:\t[%d %d %d %d]", TetNodeRank[0], TetNodeRank[1], TetNodeRank[2], TetNodeRank[3]);
			for (int n = 0; n < 4; n++)
			{
				printf("\n\tF[%d] = { %f, %f, %f }", n, F[0+3*n], F[1+3*n], F[2+3*n]);
			}
		}
#endif
}
