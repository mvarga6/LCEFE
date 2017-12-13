#ifndef __DATASTRUCT_H__
#define __DATASTRUCT_H__

#include "cuda.h"
#include "cuda_runtime.h"
#include "stdio.h"
#include "defines.h"
#include "node_array.h"
#include "tet_array.h"
#include "simulation_parameters.h"
#include "pointer.h"

///
/// Container for all data pointers that point
/// to only device (GPU) data. Difference that
/// HostDataBlock becuase we need to store the memory
/// pitches when allocation 'pitched gpu memory'
struct DevDataBlock 
{
	int Ntets, Nnodes;
	real *A;
	int *TetToNode;
	real *r0;
	real *r;
	real *F;
	real *dF;
	real *v;
	int *nodeRank;
	int *TetNodeRank;
	real *dr;
	real *m;
	real *pe;
	real *TetVol;
	int *ThPhi;
	float *S;
	int *L;
	size_t TetToNodepitch;
	size_t Apitch;
	size_t r0pitch;
	size_t rpitch;
	size_t Fpitch;
	size_t vpitch;
	size_t drpitch;
	size_t dFpitch;
	
	/// 
	/// Returns a pointer handle for the S array
	PointerHandle<real> HandleForS();
	
	/// 
	/// Returns a pointer handle for the theta-phi array
	PointerHandle<int> HandleForDirector();
};

///
/// Container for all data pointers that point
/// to host (cpu) data
class HostDataBlock 
{
public:
	int Ntets, Nnodes;
	real *A;
	int *TetToNode;
	real *r0;
	real *r;
	real *F;
	real *v;
	int *nodeRank;
	int *TetNodeRank;
	real *dr;
	real *m;
	real *pe;
	real totalVolume;
	real *TetVol;
	int *ThPhi;
	float *S;

	real min[3], max[3];
	
	///
	/// Construct with a NodeArray and TetArray (probably coming from
	/// members of Mesh) and the SimulationParameters object. 
	HostDataBlock(NodeArray *, TetArray*, SimulationParameters *);
	
	///
	/// Create a DevDataBlock with corresponding data allocations
	/// as the current HostDataBlock object
	DevDataBlock* CreateDevDataBlock();
};


#endif //__DATASTRUCT_H__
