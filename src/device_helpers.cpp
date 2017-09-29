#include "device_helpers.h"
#include "texdef.h"

__device__
void DeviceHelpers::ReadGlobalToLocal(
	int (&node_num_local)[4]
	,int (&tet_node_rank_local)[4]
	,float (&Ainv_local)[16]
	,float (&r0_local)[12]
	,float (&r_local)[12]
	,float (&v_local)[12]
	,float *Ainv_global
	,int Ashift
	,float *v_global
	,int vshift
	,int *tet_node_rank_global
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
		tet_node_rank_local[n] = tet_node_rank_global[Ntets*n + tid];
		
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
void DeviceHelpers::SendForce(
	 float *dF
	,int dFshift
	,float F[12]
	,int NodeNum[4]
	,int TetNodeRank[4]
	,float TetVol
)
{
	int n_glob, NodeRank;  

	//loop over each node in tetrahedra
	for (int n = 0; n < 4; n++)
	{ 
		//find real node number
		n_glob = NodeNum[n];    
		NodeRank = TetNodeRank[n];

		dF[dFshift*(0+3*NodeRank)+n_glob] = F[0+3*n] * TetVol;
		dF[dFshift*(1+3*NodeRank)+n_glob] = F[1+3*n] * TetVol;
		dF[dFshift*(2+3*NodeRank)+n_glob] = F[2+3*n] * TetVol;
	}//n
}
