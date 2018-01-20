#ifndef __SUMFORCE_H__
#define __SUMFORCE_H__

//#include "mainhead.h"
#include "defines.h"
#include "parameters.h"

///
/// read in a sum all forces present on node 
/// being executed on this thread
__device__ void sumForce( int myNode
						, int myNodeRank
						, real (&Fnew)[3]
						, real (&Fold)[3]
						, real (&vold)[3]
						, real *dF
						, int dFshift
						, real *F
						, int Fshift
						, real *v
						, int vshift)
{
	// for each coordinate (x,y,z)
	for (int c = 0; c < 3; c++)
	{
		//sum forces from each tet this node belongs to 
		for (int r = 0; r < myNodeRank; r++)
		{
			Fnew[c] += dF[dFshift*(c + 3*r) + myNode];

#ifdef __DEBUG_SUM_FORCE__
			if (myNode == __DEBUG_SUM_FORCE__)
			{
				printf("\n%d[%d][%d]:%f", myNode, r, c, dF[dFshift*(c + 3*r) + myNode]);
			}
#endif
		}
		
		//read in old forces
		Fold[c] = F[Fshift * c + myNode];

		//set old forces to new forces
		F[Fshift*c + myNode] = Fnew[c];

		//read in old velocities
		vold[c] = v[vshift*c + myNode];
	}
}//sumForce

#endif //__SUMFORCE_H__
