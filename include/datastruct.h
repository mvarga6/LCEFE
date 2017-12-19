#ifndef __DATASTRUCT_H__
#define __DATASTRUCT_H__

#include "cuda.h"
#include "cuda_runtime.h"
#include "stdio.h"
#include "defines.h"
#include "node_array.h"
#include "tet_array.h"
#include "tri_array.h"
#include "simulation_parameters.h"
#include "pointer.h"

class DataBlockBase
{
public:
	///
	/// # of tets, nodes, and tris 
	int Ntets, Nnodes, Ntris; 
	///
	/// Reference node positions
	real *r0;
	///
	/// Actual node positions
	real *r;
	///
	/// Summed forces on each node
	real *F;
	///
	/// Velocity of each node
	real *v;
	///
	/// Displacement of each node
	real *dr;
	///
	/// Mass of each node
	real *m;
	///
	/// Potential energy in each tet
	real *pe;
	///
	/// Shape function of each tet
	real *A;
	///
	/// Total volume of mesh
	real totalVolume;
	///
	/// Volume of each tet
	real *TetVol;
	///
	/// Area of each triangle
	real *TriArea;
	///
	/// Normal vector of each triangle
	real *TriNormal;
	///
	/// Each triangle constibutes a bit of enclosed volume
	/// (if they are surface triangles)
	real *EnclosedVolume;
	///
	/// The total volume enclosed
	real TargetEnclosedVolume;
	///
	/// Theta and Phi of director in each tet
	int *ThPhi;
	///
	/// The order parameter in each tet
	real *S;
	///
	/// Mappings (Nodes <-> Tets & Nodes <-> Tris)
	/// Could probably be its own data structure
	///
	/// Each tet has idx of its 4 nodes
	int *TetToNode;
	///
	/// Each tri has idx of its 3 nodes
	int *TriToNode;
	///
	/// Each node belongs to N tets
	int *nodeRank;
	///
	/// Each node belongs to M tris
	int *nodeRankWrtTris;
	///
	/// Each tet knows the rank of its 4 nodes
	/// (could be eliminated by using TetToNode then get NodeRank)
	int *TetNodeRank;
	///
	/// Each tri know the rank of its 3 nodes
	/// (could be eliminated vy using TriToNode then get NodeRank)
	int *TriNodeRank;
	
};

///
/// Container for all data pointers that point
/// to only device (GPU) data. Difference that
/// HostDataBlock becuase we need to store the memory
/// pitches when allocation 'pitched gpu memory'
class DevDataBlock : public DataBlockBase
{
public:
	/*
	int Ntets, Nnodes, Ntris;
	real *A;
	int *TetToNode;
	int *TriToNode;
	real *r0;
	real *r;
	real *F;
	real *v;
	int *nodeRank;
	int *TetNodeRank;
	int *TriNodeRank;
	real *dr;
	real *m;
	real *pe;
	real *TetVol;
	int *ThPhi;
	real *S; */

	///
	/// Each node stores forces on it from each 
	/// element its a member of
	real *dF;
	///
	/// (unnused) The 'Illumination' measure at each tet
	int *L;

	///
	/// The memory pitches of memory on GPU device 
	///
	size_t TetToNodepitch;
	size_t TetNodeRankpitch;
	size_t TriToNodepitch;
	size_t TriNodeRankpitch;
	size_t TriNormalpitch;
	size_t Apitch;
	size_t r0pitch;
	size_t rpitch;
	size_t Fpitch;
	size_t vpitch;
	size_t drpitch;
	size_t dFpitch;

	//real *F_tri;
	//size_t Ftripitch;
	
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
class HostDataBlock : public DataBlockBase
{
public:
	//int Ntets, Nnodes, Ntris;
	//real *A;
	//int *TetToNode;
	//int *TriToNode;
	//real *r0;
	//real *r;
	//real *F;
	//real *v;
	//int *nodeRank;
	//int *TetNodeRank;
	//int *TriNodeRank;
	//real *dr;
	//real *m;
	//real *pe;
	//real totalVolume;
	//real *TetVol;
	//real *TriArea;
	//real *TriNormal;
	//int *ThPhi;
	//real *S;

	real min[3], max[3];
	
	///
	/// Construct with a NodeArray and TetArray (probably coming from
	/// members of Mesh) and the SimulationParameters object. 
	HostDataBlock(NodeArray *, TetArray*, TriArray*, SimulationParameters *);
	
	///
	/// Create a DevDataBlock with corresponding data allocations
	/// as the current HostDataBlock object
	DevDataBlock* CreateDevDataBlock();
};


#endif //__DATASTRUCT_H__
