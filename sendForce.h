#ifndef __SENDFORCE_H__
#define __SENDFORCE_H__

#include "mainhead.h"

//===========================================================
// Send forces to linear pitched global device array to 
// later be summed and get total force one each node
//==========================================================
__device__ void sendForce(    float *dF
							, int dFshift
							, float F[12]
							, int node_num[4]
							, int TetNodeRank[4]
							, float myVol){


	int n_glob,NodeRank;  

		for (int n=0;n<4;n++){  //loop over each node in tetrahedra

			//find real node number
			n_glob = node_num[n];    
			NodeRank = TetNodeRank[n];

				dF[dFshift*(0+3*NodeRank)+n_glob]=F[0+3*n]*myVol;
				dF[dFshift*(1+3*NodeRank)+n_glob]=F[1+3*n]*myVol;
				dF[dFshift*(2+3*NodeRank)+n_glob]=F[2+3*n]*myVol;
				

		}//n

}



#endif//__SENDFORCE_H__