#ifndef __DEVICE_HELPERS_H__
#define __DEVICE_HELPERS_H__

#include "cuda.h"
#include "cuda_runtime.h"

class DeviceHelpers
{
public:
	__device__
	static void ReadGlobalToLocal(
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
	);
	
	__device__
	static void SendForce(
		 float *dF
		,int dFshift
		,float F[12]
		,int node_num[4]
		,int tet_node_rank[4]
		,float tet_vol
	);
};

#endif
