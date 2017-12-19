#ifndef __DEVICE_HELPERS_H__
#define __DEVICE_HELPERS_H__

#include "cuda.h"
#include "cuda_runtime.h"

#include "defines.h"

///
/// Service class with gpu only static methods
/// for helping on gpu
class DeviceHelpers
{
public:

	///
	/// Reads global gpu memory data into tet-thread local
	/// data arrays specific to a single tetrahedron.
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
		,int TNRshift
		,int *tet_to_node_global
		,int TTNshift
		,int Ntets
	);
	
	///
	/// Write forces on 4 nodes to global
	/// memory in the correct rank location
	/// so it can be summed in node-local thread
	static __device__
	void SendTetForce(
		 real *dF
		,int dFshift
		,real F[12]
		,int node_num[4]
		,int tet_node_rank[4]
		,real tet_vol,
		int tid
	);

	///
	/// Write forces on 3 nodes to global
	/// memory in the correct rank location
	/// so it cacn be summe din node-local thread
	static __device__
	void SendTriForce(
		real *dF
		,int dFshift
		,real F[9]
		,int node_idx[3]
		,int tri_node_rank[3]
		,int tid
	);
	
	///
	/// TODO: Should be moved to DeviceHelpers
	/// Convert two ints into a double on the gpu
	/// Will be used to support double precision
	static __inline__ __device__ 
	double ConvertToDouble(uint2 p){
    	return __hiloint2double(p.y, p.x);
	}
};

#endif
