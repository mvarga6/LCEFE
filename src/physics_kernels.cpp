#include "physics_kernels.h"
//#include "sendForce.h"
//#include "read_dev_data.h"
#include "getQ.h"
#include "device_helpers.h"
#include "sumForce.h"
#include "kernel_constants.h"
#include "update_r.h"
#include "physics_model.h"

__constant__ PackedParameters Parameters;

__global__ void ForceKernel(DevDataBlock data, real t)
{
	// ** NOTE **
	// Not using pitched memory anymore
	// so shift values are just equal to
	// to either the # of tets or nodes
	// based on what the array is.
	// Assigning the same values here so not
	// every function definition and parameters
	// need to change.
	const int Ntets = data.Ntets;
	const int Nnodes = data.Nnodes;
	const int Ashift = Ntets;
	const int dFshift = Nnodes;
	const int vshift = Nnodes;
	const int TTNshift = Ntets;
	real Ainv[16];
	real r[12];
	real r0[12];
	real F[12]={0.0};
	real vlocal[12];
	int NodeNum[4];
	int TetNodeRank[4];
	real Q[9] = {0.0};
	real myVol;

	//thread ID
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	//if thread executed has a tetrahedra
	if(tid < data.Ntets)
	{ 


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
			data.TetNodeRank,
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
		getQ(data.ThPhi[tid], Q, t, data.S[tid]); // just for debugging

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
		DeviceHelpers::SendForce(data.dF, dFshift, F, NodeNum, TetNodeRank, myVol, tid);

#ifdef __DEBUG_FORCE__
		
		// debugging info
		if (tid == __DEBUG_FORCE__)
		{
			printf("\n -- force_kernel --")
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


__global__ void UpdateKernel(DevDataBlock data)
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
	//thread ID
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid < data.Nnodes) //if a node is here
	{  
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
