#ifndef __READ_DEV_DATA_H__
#define __READ_DEV_DATA_H__

#include "parameters.h"


//get all data that will not change throughout the kernal call
// Ainv, r0 and node_num
__device__ void get_initial_data(float (&Ainv)[16]
								,float (&r0)[12]
								,int (&node_num)[4]
								,int Ashift
								,float *A
								,float *v
								,int vshift
								,float (&vlocal)[12]
								,int *TetNodeRankG
								,int (&TetNodeRank)[4]
								,int *TetToNode
								,int TTNshift
								,int Ntets){
	
	int mynode;
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
			for(int n = 0;n<4;n++){

				//figure out what 4 nodes make up tetrahedra
				mynode = TetToNode[TTNshift*n+tid];
				node_num[n] = mynode;
				TetNodeRank[n] = TetNodeRankG[tid + Ntets*n];
				for(int cord = 0;cord<4;cord++){

					if(cord<3){
					//get orignal positions
					r0[cord+n*3] = tex2D(texRef_r0,mynode,cord);
					vlocal[cord+n*3] = v[vshift*(cord+n*3)+mynode];
					}//cord<3
					//get values of Ainv
					Ainv[cord+n*4] = A[Ashift*(cord+n*4)+tid];
				}//cord
			}//n
	
}//get_initial_data



//read data that will be read per iteration
//boolian switcher will be used to know which
//texture to read from
__device__ void get_variable_data(float (&r)[12],int *node_num){

	
		for(int n = 0;n<4;n++){                   //loop over nodes
			for(int cord=0;cord<3;cord++){        //loop over x,y,z
				r[cord+n*3] = tex2D(texRef_r,node_num[n],cord);
			}//cord
		}//n
	

}




#endif//__READ_DEV_DATA_H__