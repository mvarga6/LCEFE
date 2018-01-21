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
	,real *v_global
	,int vshift
	,int *tet_node_rank_global
	,int TNRshift
	,int *tet_to_node_global
	,int TTNshift
	,int Ntets
)
{
	int node_idx;
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	for(int n = 0; n < 4; n++)
	{
		//figure out what 4 nodes make up tetrahedra
		node_idx = tet_to_node_global[TTNshift*n + tid];
		tet_node_rank_local[n] = tet_node_rank_global[TNRshift*n + tid];

#ifdef __DEBUG_READ_GLOBAL_MEMORY__
		if (tid == __DEBUG_READ_GLOBAL_MEMORY__)
		{
			printf("\n%d\tNode idx: %d Node rank: %d", tid, node_idx, tet_node_rank_local[n]);
		}
#endif

		
		node_num_local[n] = node_idx;
		
		for(int cord = 0; cord < 4; cord++)
		{
			if(cord < 3)
			{
				//get orignal positions
				r0_local[cord+n*3] = tex2D(texRef_r0, node_idx, cord);
				r_local[cord+n*3]  = tex2D(texRef_r, node_idx, cord);
				v_local[cord+n*3]  = v_global[vshift*(cord+n*3) + node_idx];
			}//cord<3
					//get values of Ainv
			Ainv_local[cord+n*4] = Ainv_global[Ashift*(cord+n*4) + tid];
		}//cord
	}//n
}
	
__device__
void DeviceHelpers::SendTetForce(
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
		for (int i = 0; i < 3; i++)
		{
			dF[dFshift*(i+3*NodeRank) + n_glob] = F[i+3*n] * TetVol;
		}
		// dF[dFshift*(0+3*NodeRank) + n_glob] = F[0+3*n] * TetVol;
		// dF[dFshift*(1+3*NodeRank) + n_glob] = F[1+3*n] * TetVol;
		// dF[dFshift*(2+3*NodeRank) + n_glob] = F[2+3*n] * TetVol;
	}
	
// #ifdef __DEBUG_SEND_FORCE__
// 		if (tid == __DEBUG_SEND_FORCE__)
// 		{
// 			printf("\n\n -- DeviceHelpers::SendForce --");
// 			printf("\n\tNode ids:\t[%d %d %d %d]", NodeNum[0], NodeNum[1], NodeNum[2], NodeNum[3]);
// 			printf("\n\tNode ranks:\t[%d %d %d %d]", TetNodeRank[0], TetNodeRank[1], TetNodeRank[2], TetNodeRank[3]);
// 			for (int n = 0; n < 4; n++)
// 			{
// 				printf("\n\tF[%d] = { %f, %f, %f }", n, F[0+3*n], F[1+3*n], F[2+3*n]);
// 			}
// 		}
// #endif
}

__device__
void DeviceHelpers::SendTriForce(
	real *dF
	,int dFshift
	,real F[9]
	,int node_idx[3]
	,int node_rank[3]
	,int tid
)
{
	int idx, NodeRank;  

	//loop over each node in triangle
	for (int n = 0; n < 3; n++)
	{ 
		//find real node number
		idx = node_idx[n];    
		NodeRank = node_rank[n];
		for (int i = 0; i < 3; i++)
		{
			//real before = dF[dFshift*(i+3*NodeRank) + idx];
			dF[dFshift*(i+3*NodeRank) + idx] += F[i+3*n];

#ifdef __DEBUG_SEND_FORCE__
			if (tid == __DEBUG_SEND_FORCE__)
			{
				printf("\n%d[%d][%d]:%f --> %f",idx, NodeRank, i, before, dF[dFshift*(i+3*NodeRank) + idx]);
			}
#endif
		}
		// dF[dFshift*(0+3*NodeRank) + n_glob] += F[0+3*n];
		// dF[dFshift*(1+3*NodeRank) + n_glob] += F[1+3*n];
		// dF[dFshift*(2+3*NodeRank) + n_glob] += F[2+3*n];
	}
	
// #ifdef __DEBUG_SEND_FORCE__
// 		if (tid == __DEBUG_SEND_FORCE__)
// 		{
// 			printf("\n\n -- DeviceHelpers::SendForce --");
// 			printf("\n\tNode ids:\t[%d %d %d]", node_idx[0], node_idx[1], node_idx[2]);
// 			printf("\n\tNode ranks:\t[%d %d %d]", node_rank[0], node_rank[1], node_rank[2]);
// 			for (int n = 0; n < 3; n++)
// 			{
// 				printf("\n\tF[%d] = { %f, %f, %f }", n, F[0+3*n], F[1+3*n], F[2+3*n]);
// 			}
// 		}
// #endif
}