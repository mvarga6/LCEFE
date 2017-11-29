#ifndef __DEVICE_HELPERS_H__
#define __DEVICE_HELPERS_H__

#include "cuda.h"
#include "cuda_runtime.h"

#include "defines.h"

class DeviceHelpers
{
public:
	static __device__
	void ReadGlobalToLocal(
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
		,int *tet_to_node_global
		,int TTNshift
		,int Ntets
	);
	
	static __device__
	void SendForce(
		 real *dF
		,int dFshift
		,real F[12]
		,int node_num[4]
		,int tet_node_rank[4]
		,real tet_vol,
		int tid
	);
	
	static __inline__ __device__ 
	double ConvertToDouble(uint2 p){
    	return __hiloint2double(p.y, p.x);
	}
};

#endif
