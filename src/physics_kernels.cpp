#include "physics_kernels.h"
//#include "sendForce.h"
//#include "read_dev_data.h"
#include "getQ.h"
#include "device_helpers.h"
#include "sumForce.h"
#include "kernel_constants.h"
#include "update_r.h"
#include "physics_model.h"
#include "texdef.h"
#include "helpers_math.h"

__constant__ PackedParameters Parameters;

__global__ void BulkForceKernel(DevDataBlock data, real t)
{
	//thread ID
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	//if thread executed has a tetrahedra
	if(tid < data.Ntets)
	{ 
		int Ashift = data.Apitch/sizeof(real);
		int dFshift = data.dFpitch/sizeof(real);
		int vshift = data.vpitch/sizeof(real);
		int TTNshift = data.TetToNodepitch/sizeof(int);
		int TNRshift = data.TetNodeRankpitch/sizeof(int);
		real Ainv[16];
		real r[12];
		real r0[12];
		real F[12]={0.0};
		real vlocal[12];
		int NodeNum[4];
		int TetNodeRank[4];
		real Q[9] = {0.0};
		real myVol;


#ifdef __DEBUG_FORCE__
		if (tid == __DEBUG_FORCE__)
		{
			printf("\n -- force_kernel --");
			printf("\n\tTTNshift: %d TNRshift: %d", TTNshift, TNRshift);
		}
#endif

		//========================================
		//read in all the data that will not change 
		//though entire simulation
		//========================================
		myVol = data.TetVol[tid];   //simple enough here

		//========================================
		//Read all the data needed for force calc
		//========================================
		DeviceHelpers::ReadGlobalToLocal(
			NodeNum, TetNodeRank,
			Ainv, r0, r, vlocal,
			data.A, Ashift,
			data.v, vshift,
			data.TetNodeRank, TNRshift,
			data.TetToNode, TTNshift,
			data.Ntets
		);
		
		//========================================
		//Calcuate Q as a function of Position
		//and time for this tetrahedra
		//
		//	-- NEW --
		// Send S and L to Q calculation and update
		// S for next calculation.
		//========================================
		//getQ(data.ThPhi[tid], Q, t, data.S[tid]); // just for debugging

		//========================================
		//calculate the force on each node due
		//to interactions in this tetrahedra
		//========================================
		Physics::CalculateForcesAndEnergies(Parameters, Ainv, r0, r, Q, F, TetNodeRank, data.pe, tid, myVol);

		//========================================
		//Send each force calculated to global 
		//memroy so force can be summed in 
		//update kernal
		//========================================
		//sendForce(data.dF, dFshift, F, NodeNum, TetNodeRank, myVol);
		DeviceHelpers::SendTetForce(data.dF, dFshift, F, NodeNum, TetNodeRank, myVol, tid);

#ifdef __DEBUG_FORCE__
		
		// debugging info
		if (tid == __DEBUG_FORCE__)
		{
			printf("\n\tTime = %f", t);
			printf("\n\tMyVol = %f", myVol);
			printf("\n\n\tF[] = ");
			printf("\n\t1: %.3f %.3f %.3f", F[0], F[1], F[2]);
			printf("\n\t2: %.3f %.3f %.3f", F[3], F[4], F[5]);
			printf("\n\t3: %.3f %.3f %.3f", F[6], F[7], F[8]);
			printf("\n\t4: %.3f %.3f %.3f", F[9], F[10], F[11]);
			printf("\n\n\tQ[] = ");
			printf("\n\t1: %.3f %.3f %.3f", Q[0], Q[1], Q[2]);
			printf("\n\t2: %.3f %.3f %.3f", Q[3], Q[4], Q[5]);
			printf("\n\t3: %.3f %.3f %.3f", Q[6], Q[7], Q[8]);
		}
#endif

	}

}


__global__ void CalculateClosedVolumesKernel(DevDataBlock data, float3 center)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < data.Ntris)
	{
		int TTNshift = data.TriToNodepitch/sizeof(int);
		int NormShift = data.TriNormalpitch/sizeof(real);
		real r[3][3]; // positions of the 3 nodes in triangle
		int node_idx[3]; // indices of the 3 nodes in triangle

		///
		/// Read global to local
		///

		// read global to local
		for (int n = 0; n < 3; n++)
		{
			node_idx[n] = data.TriToNode[TTNshift*n + tid];
			//node_rank[n] = data.TriNodeRank[TNRshift*n + tid];
			for (int d = 0; d < 3; d++)
			{
				r[n][d] = tex2D(texRef_r, node_idx[n], d);
			}
		}

		///
		/// Calculate volume made with triangle and center
		///
		real enclosedVolumeContibution =
		math::tetVolume(
			r[0][0], r[0][1], r[0][2],
			r[1][0], r[1][1], r[1][2],
			r[2][0], r[2][1], r[2][2],
			center.x, center.y, center.z
		);

		// calculate area
		real area =
		math::triangle_area(
			r[0][0], r[0][1], r[0][2],
			r[1][0], r[1][1], r[1][2],
			r[2][0], r[2][1], r[2][2]
		);
		
		// calculate normal of triangle
		real normal[3]; // normal vector of triangle
		math::triangle_normal(
			r[0][0], r[0][1], r[0][2],
			r[1][0], r[1][1], r[1][2],
			r[2][0], r[2][1], r[2][2],
			normal
		);

		///
		/// Set Global Memory
		///

		data.EnclosedVolume[tid] = enclosedVolumeContibution;
		data.TriArea[tid] = area;
		data.TriNormal[tid + 0*NormShift] = normal[0];
		data.TriNormal[tid + 1*NormShift] = normal[1];
		data.TriNormal[tid + 2*NormShift] = normal[2];
	}
}

__global__ void PressureForcesKernel(DevDataBlock data, const real V, const real V0, const real Kp)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < data.Ntris)
	{
		int TTNshift = data.TriToNodepitch/sizeof(int);
		int NormShift = data.TriNormalpitch/sizeof(real);
		int TNRshift = data.TriNodeRankpitch/sizeof(int);
		int dFshift = data.dFpitch/sizeof(real);
		real F[9] = { 0.0 }; // forces on those nodes
		
		///
		/// Pull in needed global memory
		///

		// indices of the 3 nodes in triangle
		int node_idx[3] = {
			data.TriToNode[tid + 0*TTNshift],
			data.TriToNode[tid + 1*TTNshift],
			data.TriToNode[tid + 2*TTNshift]
		};

		// rank of the nodes ( so we know where to write forces )
		int node_rank[3] = {
			data.TriNodeRank[tid + 0*TNRshift],
			data.TriNodeRank[tid + 1*TNRshift],
			data.TriNodeRank[tid + 2*TNRshift]
		};

		// area of the triangle
		real A = data.TriArea[tid];
		
		// normal of the triangle
		real normal[3] = {
			data.TriNormal[tid + 0*NormShift],
			data.TriNormal[tid + 1*NormShift],
			data.TriNormal[tid + 2*NormShift]
		};

		//
		// The Calculation of pressure forces
		//

		// The of force on tri area (should into Physics model method)
		const real _f = Kp * (V - V0) * (A / 3.0);

		// apply force evenly across 3 nodes along normal
		for (int n = 0; n < 3; n++) // for each node
		{
			for (int i = 0; i < 3; i++) // each coord
			{
				// force component is normal comp scaled by force
				F[i + n*3] = -_f * normal[i];
			}		
		}

#ifdef __DEBUG_PRESSURE__
		if (tid == __DEBUG_PRESSURE__)
		{
			printf("\n\tnode_idx:  [%d %d %d]", node_idx[0], node_idx[1], node_idx[2]);
			printf("\n\tnode_rank: [%d %d %d]", node_rank[0], node_rank[1], node_rank[2]);
			printf("\n\tnormal:    [%f.3 %f.3 %f.3]", normal[0], normal[1], normal[2]);
			printf("\n\tarea:	   %f", A);
			printf("\n\tF on node  %f", _f);
		}
#endif

		// write forces on nodes to proper
		// position in dF global memory
		DeviceHelpers::SendTriForce(
			data.dF, 
			dFshift, 
			F, 
			node_idx, 
			node_rank, 
			tid
		);
	}	
}

__global__ void UpdateKernel(DevDataBlock data)
{	
	//thread ID
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < data.Nnodes) //if a node is here
	{  
		int dFshift = data.dFpitch/sizeof(real);
		int Fshift = data.Fpitch/sizeof(real);
		int vshift = data.vpitch/sizeof(real);
		int rshift = data.rpitch/sizeof(real);
		int myNode;
		int myNodeRank;
		real Fnew[3]={0.0};
		real Fold[3];
		real vold[3];
		real vnew[3];
		real localMass;

		myNode = tid;
		myNodeRank = data.nodeRank[myNode];
		localMass = data.m[myNode];

		//get new and old forces + old velocities
		sumForce(myNode, myNodeRank, Fnew, Fold, vold, data.dF, dFshift, data.F, Fshift, data.v, vshift);

		//calculate and store new velocites
		update_v(vnew, vold, Fold, Fnew, data.v, vshift, myNode, localMass, Parameters.Dt, Parameters.Damp);

		//calculate and store new positions
		update_r(data.r, rshift, vnew, Fnew, myNode, localMass, Parameters.Dt);

	}//tid<Nnodes
}//updateKernel
